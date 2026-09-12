"""
qwen3vl.py -- GGUF key map for Qwen3-VL text/vision encoders.

This covers the unsloth quantized text encoder GGUF:
  qwen3vl_32b_minimax_h3-Q4_K_M.gguf

The unsloth GGUF uses a slightly different key naming convention from the
HuggingFace safetensors checkpoint:

  GGUF key                               → Diffusers/HF checkpoint key
  ──────────────────────────────────────────────────────────────────────
  model.embed_tokens.weight              → model.language_model.embed_tokens.weight
  model.layers.{i}.*                     → model.language_model.layers.{i}.*
  visual.blocks.{i}.*                    → model.visual.blocks.{i}.*
  visual.merger.*                        → model.visual.merger.*
  visual.patch_embed.*                   → model.visual.patch_embed.*
  visual.pos_embed.*                     → model.visual.pos_embed.*
  visual.deepstack_merger_list.*         → model.visual.deepstack_merger_list.*

Detection: keys starting with "visual.blocks." (no "model." prefix on visual)
           AND "model.layers." (no "language_model." sub-prefix on LM layers).
"""

import logging
from typing import Any, Dict, List

import torch
from accelerate.utils.modeling import set_module_tensor_to_device

logger = logging.getLogger("weellm")


class Qwen3VLKeyMap:
    NAME = "qwen3vl"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # Detect by absence of model.language_model prefix but presence of model.layers
        has_model_layers = any(k.startswith("model.layers.") for k in gguf_keys)
        has_visual_no_model_prefix = any(k.startswith("visual.blocks.") for k in gguf_keys)
        has_no_language_model = not any(k.startswith("model.language_model.") for k in gguf_keys)
        return has_model_layers and has_visual_no_model_prefix and has_no_language_model

    @staticmethod
    def _remap_key(gguf_key: str) -> str:
        """Translate a single GGUF key to HF checkpoint naming convention."""
        # model.embed_tokens → model.language_model.embed_tokens
        if gguf_key.startswith("model.embed_tokens."):
            return "model.language_model.embed_tokens." + gguf_key[len("model.embed_tokens."):]

        # model.layers.{i}.* → model.language_model.layers.{i}.*
        if gguf_key.startswith("model.layers."):
            return "model.language_model.layers." + gguf_key[len("model.layers."):]

        # visual.* → model.visual.*
        if gguf_key.startswith("visual."):
            return "model." + gguf_key

        return gguf_key

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for gguf_key in gguf_keys:
            mapped_key = Qwen3VLKeyMap._remap_key(gguf_key)
            remap[gguf_key] = [(mapped_key, None)]
        return remap

    @staticmethod
    def patch_model_before_stream(model: torch.nn.Module, cfg, device: str, dtype: torch.dtype, seeker=None) -> None:
        """
        Initialise tensors that are omitted from the GGUF weights because they are either
        analytically derived (RoPE inv_freq) or tied (lm_head <-> embed_tokens).
        """
        import math
        lm_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg

        def _init_rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
            half = head_dim // 2
            return 1.0 / (
                theta ** (torch.arange(0, half, dtype=torch.float32) / half)
            )

        # 1. Language-model final norm (RMSNorm) - initialise to ones
        try:
            norm = model.model.language_model.norm
            if next(norm.parameters()).device.type == "meta":
                ones = torch.ones(lm_cfg.hidden_size, dtype=dtype, device=device)
                set_module_tensor_to_device(model, "model.language_model.norm.weight",
                                            device, value=ones)
                logger.info("  [GGUF fix] Initialised model.language_model.norm.weight to ones.")
        except (AttributeError, StopIteration):
            pass

        # 2. Language-model RoPE buffers
        try:
            lm_rope = model.model.language_model.rotary_emb
            head_dim = getattr(lm_cfg, "head_dim",
                               lm_cfg.hidden_size // lm_cfg.num_attention_heads)
            theta   = getattr(lm_cfg, "rope_theta", 1_000_000.0)
            inv_freq = _init_rope_inv_freq(head_dim, theta).to(device)
            for attr in ("inv_freq", "original_inv_freq"):
                buf = getattr(lm_rope, attr, None)
                if buf is not None and buf.device.type == "meta":
                    set_module_tensor_to_device(
                        model,
                        f"model.language_model.rotary_emb.{attr}",
                        device, value=inv_freq.clone(),
                    )
                    logger.info("  [GGUF fix] Initialised model.language_model.rotary_emb.%s.", attr)
        except AttributeError:
            pass

        # 3. Vision RoPE buffer
        try:
            vis_rope = model.model.visual.rotary_pos_emb
            vis_cfg  = cfg.vision_config if hasattr(cfg, "vision_config") else cfg
            head_dim = getattr(vis_cfg, "hidden_size", 1152) // getattr(vis_cfg, "num_heads", 16)
            theta    = getattr(vis_cfg, "rope_theta", 10_000.0)
            inv_freq = _init_rope_inv_freq(head_dim, theta).to(device)
            buf = getattr(vis_rope, "inv_freq", None)
            if buf is not None and buf.device.type == "meta":
                set_module_tensor_to_device(
                    model, "model.visual.rotary_pos_emb.inv_freq",
                    device, value=inv_freq,
                )
                logger.info("  [GGUF fix] Initialised model.visual.rotary_pos_emb.inv_freq.")
        except AttributeError:
            pass

        # 4. lm_head: tie to embed_tokens if configured
        try:
            lm_head = model.lm_head
            tie = getattr(cfg, "tie_word_embeddings", True)
            if tie:
                embed_w = model.model.language_model.embed_tokens.weight
                if lm_head.weight.device.type == "meta" and embed_w.device.type != "meta":
                    set_module_tensor_to_device(
                        model, "lm_head.weight",
                        device, value=embed_w.data,
                    )
                    logger.info("  [GGUF fix] Tied lm_head.weight to embed_tokens.weight.")
        except AttributeError:
            pass

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: torch.Tensor, orig_name: str) -> torch.Tensor:
        """Fix shape mismatches caused by GGUF flattening multi-dim tensors."""
        # Qwen3-VL patch_embed weight from Unsloth GGUF needs to be reshaped
        # GGUF stores as [out*in, t, h, w], model expects [out, in, t, h, w]
        if diffusers_key == "model.visual.patch_embed.proj.weight" and tensor.shape == (3456, 2, 16, 16):
            return tensor.view(1152, 3, 2, 16, 16)
        return tensor