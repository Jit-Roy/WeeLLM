"""
qwen3_5_model.py -- Hook-based layer-streaming for Qwen3_5Model (MOSS-Image text encoder).

Streams the massive 9B parameter language model directly from original safetensors shards
via live seek, keeping under 4GB VRAM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModel

from weellm.io.seeker import get_seeker
from weellm.io.memory import pin_module_to_cpu, place_tensors
from weellm.io.utils import clean_memory, default_dtype

logger = logging.getLogger("weellm")


def _has_module_path(model: nn.Module, name: str) -> bool:
    current = model
    for part in name.split(".")[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
        if current is None:
            return False
    return hasattr(current, name.split(".")[-1])


def _map_key(model: nn.Module, name: str) -> Optional[str]:
    if _has_module_path(model, name):
        return name
    if name.startswith("model.") and _has_module_path(model, name[6:]):
        return name[6:]
    return None


def _get_resident_keys(seeker) -> List[str]:
    # Everything that is not a layer block is resident, excluding lm_head/mtp if present
    keys = []
    for k in seeker.weight_map.keys():
        if "lm_head" in k or "mtp" in k:
            continue
        if not ("layers." in k or "encoder.layers" in k or "visual.blocks." in k):
            keys.append(k)
    return keys


class Qwen3_5ModelStreamer:
    """
    Hook-based streaming text encoder for Qwen3_5Model.
    Streams layers dynamically from safetensors shards.
    """

    def __init__(
        self,
        text_encoder_dir: Path | str,
        tokenizer=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        **kwargs,
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.tokenizer = tokenizer
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
        logger.info("Initializing streaming Qwen3_5Model text encoder ...")
        self._seeker = get_seeker(self.text_encoder_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_resident_modules()
        self._install_hooks()

        self._initialized = True
        logger.info("Qwen3_5Model text encoder ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
        self._model.eval()

        # Monkey-patch .to() to prevent diffusers from crashing on meta tensors
        def _noop_to(self, *args, **kwargs):
            return self
        self._model.__class__.to = _noop_to

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)

        cpu_sd = {k: v for k, v in resident_sd.items() if "embed_tokens" in k}
        gpu_sd = {k: v for k, v in resident_sd.items() if k not in cpu_sd}

        if cpu_sd:
            self._place_tensors(cpu_sd, device="cpu")
            if hasattr(self._model, "language_model") and hasattr(self._model.language_model, "embed_tokens"):
                pin_module_to_cpu(self._model, "language_model.embed_tokens")
            elif hasattr(self._model, "embed_tokens"):
                pin_module_to_cpu(self._model, "embed_tokens")

        if gpu_sd:
            self._place_tensors(gpu_sd, device=self.device)

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
        for raw_name, tensor in state_dict.items():
            name = _map_key(self._model, raw_name)
            if name is None:
                continue
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self._model, name, device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self._model, name, device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for raw_name in state_dict.keys():
            name = _map_key(self._model, raw_name)
            if name is not None:
                set_module_tensor_to_device(self._model, name, "meta")

    def _install_hooks(self):
        # Force any tensor inputs to execution device
        def force_inputs_to_device(module, args, kwargs):
            new_args = tuple(arg.to(self.device) if torch.is_tensor(arg) else arg for arg in args)
            new_kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return new_args, new_kwargs

        self._model.register_forward_pre_hook(force_inputs_to_device, with_kwargs=True)

        # Hook Language Layers
        lang_prefix = "language_model"
        if hasattr(self._model, "language_model") and hasattr(self._model.language_model, "layers"):
            lang_layers = self._model.language_model.layers
            for i in range(len(lang_layers)):
                layer = lang_layers[i]
                layer._te_prefix = f"{lang_prefix}.layers.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)

        # Hook Visual Blocks if present
        if hasattr(self._model, "visual") and hasattr(self._model.visual, "blocks"):
            vis_blocks = self._model.visual.blocks
            for i in range(len(vis_blocks)):
                layer = vis_blocks[i]
                layer._te_prefix = f"visual.blocks.{i}."
                layer.register_forward_pre_hook(self._generic_pre_hook)
                layer.register_forward_hook(self._generic_post_hook)

    def _generic_pre_hook(self, module: nn.Module, args):
        prefix = module._te_prefix
        layer_keys = [
            k for k in self._seeker.weight_map.keys()
            if k.startswith(prefix) or k.startswith("model." + prefix)
        ]
        gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)
        self._place_tensors(gpu_sd)
        module._te_loaded_sd = gpu_sd

    def _generic_post_hook(self, module: nn.Module, args, output):
        loaded_sd = getattr(module, "_te_loaded_sd", None)
        if loaded_sd is not None:
            self._evict_layer(loaded_sd)
            module._te_loaded_sd = None
        return output

    @property
    def model(self):
        return self._model

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        tokenizer=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        **kwargs,
    ) -> "Qwen3_5ModelStreamer":
        return cls(
            text_encoder_dir=model_dir,
            tokenizer=tokenizer,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
        )
