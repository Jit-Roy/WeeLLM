"""
minimax_h3_qwen3_vl_hf_encoder.py -- Hook-based layer-streaming for the MiniMaxH3Qwen3VLHFEncoder.

Uses single-stream live buffering to stream both language layers and vision blocks
directly from the SSD to prevent OOM on the massive weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from weellm.utils import default_dtype
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModel

from weellm.utils import clean_memory
from weellm.seeker import get_seeker


def _get_resident_keys(seeker) -> List[str]:
    # Everything that is not a layer block is resident, EXCLUDING the lm_head
    # which is unused in diffusion models but takes a lot of RAM.
    keys = []
    for k in seeker.weight_map.keys():
        if "lm_head" in k:
            continue
        if not ("layers." in k or "encoder.layers" in k or "visual.blocks." in k):
            keys.append(k)
    return keys


class MiniMaxH3Qwen3VLHFEncoderStreamer:
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
        print("Initialising streaming MiniMax-H3 Qwen3VL text encoder ...")
        self._seeker = get_seeker(self.text_encoder_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_resident_modules()
        self._install_hooks()
        
        self._initialized = True
        print("MiniMax-H3 Qwen3VL text encoder ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
        self._model.eval()
        
        # Monkey-patch .to() to prevent diffusers from crashing on meta tensors
        def _noop_to(self_obj, *args, **kwargs):
            return self_obj
        self._model.__class__.to = _noop_to

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        
        cpu_sd = {k: v for k, v in resident_sd.items() if "embed_tokens" in k}
        gpu_sd = {k: v for k, v in resident_sd.items() if k not in cpu_sd}
        
        if cpu_sd:
            self._place_tensors(cpu_sd, device="cpu")
            from weellm.memory import pin_module_to_cpu
            if hasattr(self._model, "language_model") and hasattr(self._model.language_model, "embed_tokens"):
                pin_module_to_cpu(self._model, "language_model.embed_tokens")
            elif hasattr(self._model, "embed_tokens"):
                pin_module_to_cpu(self._model, "embed_tokens")
                
        if gpu_sd:
            self._place_tensors(gpu_sd, device=self.device)
            
        del resident_sd, cpu_sd, gpu_sd

        # Handle rotary embeddings if present
        if hasattr(self._model, "language_model") and hasattr(self._model.language_model, "rotary_emb"):
            rotary = self._model.language_model.rotary_emb
            for buf_name, buf in list(rotary.named_buffers()):
                if buf.device.type != self.device:
                    set_module_tensor_to_device(
                        self._model, f"language_model.rotary_emb.{buf_name}",
                        self.device, value=buf.float()
                    )

        clean_memory(self.device)

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor], device: Optional[str] = None):
        device = device or self.device
        processed_sd = {}
        for name, tensor in state_dict.items():
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
        if hasattr(self._model, "language_model") and hasattr(self._model.language_model, "layers"):
            lang_layers = self._model.language_model.layers
            for i in range(len(lang_layers)):
                layer = lang_layers[i]
                layer._te_prefix = f"language_model.layers.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)
        
        # Hook Visual Blocks
        if hasattr(self._model, "visual") and hasattr(self._model.visual, "blocks"):
            vis_blocks = self._model.visual.blocks
            for i in range(len(vis_blocks)):
                layer = vis_blocks[i]
                layer._te_prefix = f"visual.blocks.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)

    def _generic_pre_hook(self, module: nn.Module, args):
        prefix = module._te_prefix
        layer_keys = [k for k in self._seeker.weight_map.keys() if k.startswith(prefix)]
        gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)
        self._place_tensors(gpu_sd)
        module._te_loaded_sd = gpu_sd
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
    ) -> "MiniMaxH3Qwen3VLHFEncoderStreamer":
        return cls(
            text_encoder_dir=model_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            **kwargs
        )
