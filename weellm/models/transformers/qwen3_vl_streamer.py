"""
qwen3_vl_streamer.py -- Hook-based layer-streaming for Qwen3VLForConditionalGeneration.

Architecture:
  - 64 language model layers
  - 27 vision blocks
  - 62GB parameters

Strategy:
  - Resident on GPU/RAM: embed_tokens, lm_head, patch_embed, pos_embed
  - Streamed: vision blocks and language model layers (loaded just-in-time via hooks)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLConfig

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

class Qwen3VLStreamer(BaseTransformerStreamer):
    """
    Streams Qwen3-VL layer-by-layer to allow inference on systems with limited VRAM.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        """
        Stream vision blocks followed by language model layers.

        NOTE: Qwen3VLForConditionalGeneration has the structure:
          model (outer)  ->  .model (inner Qwen3VLModel)  ->  .visual / .language_model
        Checkpoint keys however use the inner prefix: model.visual.blocks.X
        """
        order = []
        inner = self.model.model  # Qwen3VLModel lives here

        # 1. Vision Blocks
        if hasattr(inner, "visual") and hasattr(inner.visual, "blocks"):
            for i, block in enumerate(inner.visual.blocks):
                # shard_name matches the checkpoint key prefix exactly
                shard_name = f"model.visual.blocks.{i}"
                order.append((shard_name, block))

        # 2. Language Model Layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            for i, layer in enumerate(inner.language_model.layers):
                shard_name = f"model.language_model.layers.{i}"
                order.append((shard_name, layer))

        return order

    def _get_resident_keys(self) -> List[str]:
        """
        Keep everything resident EXCEPT the streamed vision blocks and LM layers.
        """
        streaming_prefixes = (
            "model.visual.blocks.",
            "model.language_model.layers."
        )
        resident_keys = []
        for k in self.seeker.weight_map.keys():
            if not any(k.startswith(p) for p in streaming_prefixes):
                resident_keys.append(k)
        return resident_keys

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        """
        Override: shard_name IS the checkpoint key prefix (e.g. 'model.visual.blocks.0').
        Checkpoint keys are like 'model.visual.blocks.0.attn.qkv.weight'.
        """
        return [
            k for k in self.seeker.weight_map
            if k.startswith(shard_name + ".")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        """
        Override: checkpoint keys are model.visual.blocks.X.* and model.language_model.layers.X.*
        but the nn.Module parameter paths (relative to self.model) include the extra outer 'model.' prefix:
          e.g. checkpoint key 'model.visual.blocks.0.attn.qkv.weight'
               maps to param   'model.visual.blocks.0.attn.qkv.weight' (same! - no remapping needed)
        So we just delegate directly.
        """
        from weellm.memory import place_tensors
        place_tensors(self.model, state_dict, self.device, self.dtype, skip_errors=skip_errors)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> Qwen3VLStreamer:
        
        model_dir = Path(model_dir)
        logger.info("Step 1/3 -- Initializing LiveSeeker on Qwen3-VL weights ...")
        seeker = get_seeker(
            str(model_dir),
            cache_to_ram=cache_to_ram
        )
        
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("  Instantiating Qwen3VLForConditionalGeneration on meta device ...")
        cfg = Qwen3VLConfig.from_pretrained(str(model_dir))
        
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(cfg)
        model.eval()

        # Transformers specific: Ensure buffers (like rotary pos emb) are moved off meta
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        logger.info("Step 2/3 -- Hooking streaming layers ...")
        streamer = cls(
            model=model,
            seeker=seeker,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )

        logger.info("Step 3/3 -- Loading resident Qwen3-VL tensors to RAM/device ...")
        resident_keys = streamer._get_resident_keys()
        logger.info("  Loading %d resident tensors (vocab, heads, norm) ...", len(resident_keys))
        
        # Load resident keys from seeker
        # We can try to load them to device directly, or keep some in RAM if they are too big.
        # Let's load them to device (if VRAM OOMs we can refine this)
        sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(sd, skip_errors=False)
        del sd
        clean_memory()
        
        logger.info("Installed %d blocks for streaming.", len(streamer._shard_order))
        logger.info("Qwen3VLStreamer ready.")
        report_memory(device)
        return streamer
