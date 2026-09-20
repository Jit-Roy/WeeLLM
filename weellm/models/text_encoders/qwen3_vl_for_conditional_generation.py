"""
qwen3_vl_for_conditional_generation.py -- Hook-based layer-streaming for Qwen3-VL text encoders.

Used by MiniMax-H3 (which ships a pruned Qwen3-VL) and by Qwen-Image 2.1 (which uses a
full Qwen3-VL checkpoint as its joint text/vision encoder).

Uses single-stream live buffering to stream both language layers and vision blocks
directly from the SSD to prevent OOM on the massive weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from weellm.io.utils import default_dtype
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from weellm.io.utils import clean_memory
from weellm.io.seeker import get_seeker

logger = logging.getLogger("weellm")


class _PassThroughLMHead(nn.Module):
    """Stand-in for an unused ``lm_head``.

    ``Qwen3VLForConditionalGeneration.forward`` always projects the final hidden states
    through ``lm_head``, but the diffusion pipelines only ever read
    ``outputs.hidden_states``. Since ``lm_head`` is excluded from the resident set (it is
    over a gigabyte on a full Qwen3-VL checkpoint) its weight stays on the meta device —
    and because it has ``bias=False`` that projection does not raise, it just returns an
    uninitialised ``(batch, seq_len, vocab_size)`` tensor. Passing the hidden states
    straight through avoids both the garbage and the allocation.
    """

    def forward(self, hidden_states):  # noqa: D102 - trivial pass-through
        return hidden_states


def _vram_gb() -> float:
    """Allocated VRAM in GB, or 0.0 when there is no CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


def _get_resident_keys(seeker) -> List[str]:
    # Everything that is not a layer block is resident, EXCLUDING the lm_head
    # which is unused in diffusion models but takes a lot of RAM.
    keys = []
    for k in seeker.weight_map.keys():
        if "lm_head" in k or "language_model.norm" in k:
            continue
        if not ("layers." in k or "encoder.layers" in k or "visual.blocks." in k):
            keys.append(k)
    return keys


class Qwen3VLForConditionalGenerationStreamer:
    """
    Hook-based streaming text encoder for MiniMax-H3 (Qwen3VL-based).
    """

    def __init__(
        self,
        text_encoder_dir: Path | str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        **kwargs
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.device = device
        self.dtype = dtype
        self.cache_to_ram = cache_to_ram

        self._seeker: Optional[object] = None
        self._model: Optional[nn.Module] = None
        self._initialized = False
        
        self._ensure_initialized()

    def _ensure_initialized(self):
        if self._initialized:
            return
        logger.info("Initialising streaming Qwen3-VL text encoder ...")
        self._seeker = get_seeker(self.text_encoder_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_resident_modules()
        self._install_hooks()

        self._initialized = True
        logger.info("Qwen3-VL text encoder ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        with default_dtype(self.dtype), init_empty_weights():
            self._model = Qwen3VLForConditionalGeneration(config)
        self._model.eval()

        # Truncate layers to match the available weights (e.g., for pruned models).
        # We leave the config layer count intact so Diffusers validation passes,
        # but truncating the module list gracefully halts the forward loop early.
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model") and hasattr(self._model.model.language_model, "layers"):
            layer_indices = [
                int(k.split("layers.")[1].split(".")[0])
                for k in self._seeker.weight_map.keys()
                if "layers." in k
            ]
            # No layer keys at all means we cannot infer anything — leave the module
            # list alone rather than truncating to an arbitrary guess.
            if layer_indices:
                max_layer = max(layer_indices)
                if max_layer + 1 < len(self._model.model.language_model.layers):
                    logger.info(
                        "[WeeLLM] Truncating language_model.layers to %d to prevent meta crash.",
                        max_layer + 1,
                    )
                    self._model.model.language_model.layers = self._model.model.language_model.layers[:max_layer + 1]

            # The final norm is not streamed. Pruned checkpoints (MiniMax-H3) ship no
            # weight for it and read a pre-norm hidden state anyway, so it is replaced
            # with Identity to prevent a meta crash. A full checkpoint (Qwen-Image 2.1)
            # does carry `norm.weight`, which lands in the resident set — replacing the
            # module there would make placing that tensor fail, so keep the real one.
            if hasattr(self._model.model.language_model, "norm"):
                norm_keys = [
                    k for k in self._seeker.weight_map
                    if k.endswith("language_model.norm.weight") or k == "model.norm.weight"
                ]
                if not norm_keys:
                    self._model.model.language_model.norm = nn.Identity()
                    logger.debug("[WeeLLM] No final-norm weight in checkpoint — replaced norm with Identity.")
                else:
                    logger.debug("[WeeLLM] Final-norm weight present (%s) — keeping the real module.", norm_keys[0])

        # lm_head is deliberately left out of the resident set, so its weight would stay
        # on the meta device while forward() still projects through it. See _PassThroughLMHead.
        if hasattr(self._model, "lm_head"):
            lm_head_is_resident = any("lm_head" in k for k in _get_resident_keys(self._seeker))
            if not lm_head_is_resident:
                self._model.lm_head = _PassThroughLMHead().eval()
                logger.debug("[WeeLLM] lm_head is not resident — replaced with a pass-through.")

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        logger.debug("[WeeLLM] VRAM before resident get_tensors: %.3f GB", _vram_gb())
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        logger.debug("[WeeLLM] VRAM after resident get_tensors: %.3f GB", _vram_gb())

        cpu_sd = {k: v for k, v in resident_sd.items() if "embed_tokens" in k}
        gpu_sd = {k: v for k, v in resident_sd.items() if k not in cpu_sd}

        logger.debug("----------------- QWEN3-VL RESIDENT TENSORS -----------------")
        gpu_bytes = 0
        for k, v in gpu_sd.items():
            mb = (v.numel() * v.element_size()) / 1024**2
            logger.debug("GPU resident tensor: %s | shape: %s | size: %.2f MB", k, list(v.shape), mb)
            gpu_bytes += mb
        logger.debug("TOTAL GPU RESIDENT: %.2f MB", gpu_bytes)

        cpu_bytes = 0
        for k, v in cpu_sd.items():
            mb = (v.numel() * v.element_size()) / 1024**2
            logger.debug("CPU resident tensor: %s | shape: %s | size: %.2f MB", k, list(v.shape), mb)
            cpu_bytes += mb
        logger.debug("TOTAL CPU RESIDENT: %.2f MB", cpu_bytes)
        logger.debug("-------------------------------------------------------------")

        if cpu_sd:
            self._place_tensors(cpu_sd, device="cpu")
            from weellm.io.memory import pin_module_to_cpu
            if hasattr(self._model, "model") and hasattr(self._model.model, "language_model") and hasattr(self._model.model.language_model, "embed_tokens"):
                pin_module_to_cpu(self._model, "model.language_model.embed_tokens")
            elif hasattr(self._model, "embed_tokens"):
                pin_module_to_cpu(self._model, "embed_tokens")
                
        if gpu_sd:
            logger.debug("[WeeLLM] VRAM before resident placement: %.3f GB", _vram_gb())
            self._place_tensors(gpu_sd, device=self.device)
            logger.debug("[WeeLLM] VRAM after resident placement: %.3f GB", _vram_gb())

        del resident_sd, cpu_sd, gpu_sd

        # Handle rotary embeddings if present
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model") and hasattr(self._model.model.language_model, "rotary_emb"):
            rotary = self._model.model.language_model.rotary_emb
            for buf_name, buf in list(rotary.named_buffers()):
                if buf.device.type != self.device:
                    set_module_tensor_to_device(
                        self._model, f"model.language_model.rotary_emb.{buf_name}",
                        self.device, value=buf.float()
                    )

        self._init_missing_tensors()
        clean_memory(self.device)

    def _init_missing_tensors(self):
        """Initialize omitted tensors (like RoPE inv_freq and tied heads) if they are still on meta."""
        cfg = self._model.config
        lm_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg

        def _init_rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
            half = head_dim // 2
            return 1.0 / (
                theta ** (torch.arange(0, half, dtype=torch.float32) / half)
            )

        # 1. Language-model final norm (RMSNorm) ─ initialise to ones
        try:
            norm = self._model.model.language_model.norm
            if next(norm.parameters()).device.type == "meta":
                ones = torch.ones(lm_cfg.hidden_size, dtype=self.dtype, device=self.device)
                set_module_tensor_to_device(self._model, "model.language_model.norm.weight",
                                            self.device, value=ones)
        except (AttributeError, StopIteration):
            pass

        # 2. Language-model RoPE buffers
        try:
            lm_rope = self._model.model.language_model.rotary_emb
            head_dim = getattr(lm_cfg, "head_dim",
                               lm_cfg.hidden_size // lm_cfg.num_attention_heads)
            theta   = getattr(lm_cfg, "rope_theta", 1_000_000.0)
            inv_freq = _init_rope_inv_freq(head_dim, theta).to(self.device)
            for attr in ("inv_freq", "original_inv_freq"):
                buf = getattr(lm_rope, attr, None)
                if buf is not None and buf.device.type == "meta":
                    set_module_tensor_to_device(
                        self._model,
                        f"model.language_model.rotary_emb.{attr}",
                        self.device, value=inv_freq.clone(),
                    )
        except AttributeError:
            pass

        # 3. Vision RoPE buffer
        try:
            vis_rope = self._model.model.visual.rotary_pos_emb
            vis_cfg  = cfg.vision_config if hasattr(cfg, "vision_config") else cfg
            head_dim = getattr(vis_cfg, "hidden_size", 1152) // getattr(vis_cfg, "num_heads", 16)
            theta    = getattr(vis_cfg, "rope_theta", 10_000.0)
            inv_freq = _init_rope_inv_freq(head_dim, theta).to(self.device)
            buf = getattr(vis_rope, "inv_freq", None)
            if buf is not None and buf.device.type == "meta":
                set_module_tensor_to_device(
                    self._model, "model.visual.rotary_pos_emb.inv_freq",
                    self.device, value=inv_freq,
                )
        except AttributeError:
            pass

        # 4. lm_head: tie to embed_tokens if configured
        try:
            lm_head = self._model.lm_head
            tie = getattr(cfg, "tie_word_embeddings", True)
            if tie:
                embed_w = self._model.model.language_model.embed_tokens.weight
                if lm_head.weight.device.type == "meta" and embed_w.device.type != "meta":
                    set_module_tensor_to_device(
                        self._model, "lm_head.weight",
                        self.device, value=embed_w.data,
                    )
        except AttributeError:
            pass

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor], device: Optional[str] = None):
        device = device or self.device
        processed_sd = {}
        for name, tensor in state_dict.items():
            # Fix patch_embed flattened by GGUF (from 5D to 4D)
            if "patch_embed.proj.weight" in name and tensor.dim() == 4:
                try:
                    param = self._model.get_parameter(name)
                    if param is not None and param.shape != tensor.shape:
                        tensor = tensor.reshape(param.shape)
                except Exception:
                    try:
                        param = dict(self._model.named_parameters()).get(name)
                        if param is not None and param.shape != tensor.shape:
                            tensor = tensor.reshape(param.shape)
                    except Exception:
                        pass
                        
            if name.endswith(".weight_scale"):
                continue
            
            if name.endswith(".weight") and f"{name}_scale" in state_dict:
                scale = state_dict[f"{name}_scale"].to(device=tensor.device, dtype=torch.float32)
                
                if scale.dim() == 1:
                    if scale.numel() == tensor.shape[0]:
                        scale = scale.view(-1, 1)
                    elif len(tensor.shape) > 1 and scale.numel() == tensor.shape[1]:
                        scale = scale.view(1, -1)
                        
                tensor = (tensor.to(torch.float32) * scale).to(self.dtype)
                
            processed_sd[name] = tensor

        for name, tensor in processed_sd.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self._model, name, device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self._model, name, device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            if name.endswith(".weight_scale"):
                continue
            set_module_tensor_to_device(self._model, name, "meta")


    def _install_hooks(self):
        # Hook Language Layers
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model") and hasattr(self._model.model.language_model, "layers"):
            lang_layers = self._model.model.language_model.layers
            for i in range(len(lang_layers)):
                layer = lang_layers[i]
                layer._te_prefix = f"model.language_model.layers.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)
        
        # Hook Visual Blocks
        if hasattr(self._model, "model") and hasattr(self._model.model, "visual") and hasattr(self._model.model.visual, "blocks"):
            vis_blocks = self._model.model.visual.blocks
            for i in range(len(vis_blocks)):
                layer = vis_blocks[i]
                layer._te_prefix = f"model.visual.blocks.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)

    def _generic_pre_hook(self, module: nn.Module, args):
        prefix = getattr(module, "_te_prefix", "")
        layer_keys = [k for k in self._seeker.weight_map.keys() if k.startswith(prefix)]
        if not layer_keys:
            # Gracefully skip missing layers in pruned GGUF models
            return args
        
        gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)
        logger.debug("[Hook] Loading %d tensors for %s onto %s", len(gpu_sd), prefix, self.device)
        if not gpu_sd:
            logger.warning("[Hook] No tensors found for %s!", prefix)
        self._place_tensors(gpu_sd)
        module._te_loaded_sd = gpu_sd  # keep original keys for eviction
        return args

    def _generic_post_hook(self, module: nn.Module, args, output):
        loaded_sd = getattr(module, "_te_loaded_sd", None)
        if loaded_sd is not None:
            self._evict_layer(loaded_sd)
            module._te_loaded_sd = None
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        **kwargs,
    ) -> "Qwen3VLForConditionalGenerationStreamer":
        return cls(
            text_encoder_dir=model_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            **kwargs
        )
