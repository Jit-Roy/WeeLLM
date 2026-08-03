"""
lumina2_transformer_2d_model.py -- Hook-based streaming for Lumina2Transformer2DModel.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel
from weellm.seeker import get_seeker
from weellm.utils import clean_memory


def _get_layer_keys(seeker, layer_idx: str) -> List[str]:
    prefix = f"{layer_idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]

def _get_resident_keys(seeker) -> List[str]:
    return [
        k for k in seeker.weight_map.keys()
        if not k.startswith("context_refiner.")
        and not k.startswith("noise_refiner.")
        and not k.startswith("layers.")
    ]


class Lumina2Transformer2DModelStreamer(nn.Module):
    def __init__(
        self,
        model_dir: Path | str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.cache_to_ram = cache_to_ram
        self.prefetch = prefetch

        self._seeker = get_seeker(self.model_dir, cache_to_ram=self.cache_to_ram)
        self.model: Optional[Lumina2Transformer2DModel] = None

        self._executor = None
        self._next_future = None
        self._next_future_idx = None
        self._lock = threading.Lock()

        self._hook_handles = []
        self.layer_sequence = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        with init_empty_weights():
            self.model = Lumina2Transformer2DModel.from_config(
                str(self.model_dir), trust_remote_code=True
            )
        self.model.eval()

        self.layer_sequence = (
            [f"context_refiner.{i}" for i in range(len(self.model.context_refiner))] +
            [f"noise_refiner.{i}" for i in range(len(self.model.noise_refiner))] +
            [f"layers.{i}" for i in range(len(self.model.layers))]
        )

        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        self._place_tensors(resident_sd)
        del resident_sd

        for buf_name, buf in list(self.model.named_buffers()):
            if buf.device.type != self.device:
                set_module_tensor_to_device(
                    self.model, buf_name, self.device, value=buf.to(self.dtype) if buf.is_floating_point() else buf
                )

        clean_memory(self.device)

        if self.prefetch:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumina_gpu_load")

        self._install_hooks()
        self._initialized = True
        print(f"Installed {len(self.layer_sequence)} transformer blocks for streaming.")

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor]):
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self.model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self.model, name, self.device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            set_module_tensor_to_device(self.model, name, "meta")

    def _install_hooks(self):
        for layer_idx_str in self.layer_sequence:
            parts = layer_idx_str.split(".")
            layer = getattr(self.model, parts[0])[int(parts[1])]
            layer._te_layer_idx = layer_idx_str

            h_pre = layer.register_forward_pre_hook(self._layer_pre_hook)
            h_post = layer.register_forward_hook(self._layer_post_hook)
            self._hook_handles.extend([h_pre, h_post])

    def _layer_pre_hook(self, module: nn.Module, args):
        idx_str = module._te_layer_idx
        layer_keys = _get_layer_keys(self._seeker, idx_str)

        with self._lock:
            if self._next_future_idx == idx_str and self._next_future is not None:
                gpu_sd = self._next_future.result()
                self._next_future = None
                self._next_future_idx = None
            else:
                gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        self._place_tensors(gpu_sd)
        module._te_loaded_sd = gpu_sd

        current_list_idx = self.layer_sequence.index(idx_str)
        next_list_idx = current_list_idx + 1

        if next_list_idx < len(self.layer_sequence) and self._executor is not None:
            next_idx_str = self.layer_sequence[next_list_idx]
            next_layer_keys = _get_layer_keys(self._seeker, next_idx_str)
            with self._lock:
                self._next_future = self._executor.submit(
                    self._seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_future_idx = next_idx_str

    def _layer_post_hook(self, module: nn.Module, args, output):
        self._evict_layer(getattr(module, "_te_loaded_sd", {}))
        module._te_loaded_sd = {}
        return output

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self._ensure_initialized()
        
        # Reset prefetch state
        with self._lock:
            if self._next_future is not None:
                self._next_future.result()
            self._next_future = None
            self._next_future_idx = None

            if self._executor is not None and len(self.layer_sequence) > 0:
                first_layer_idx = self.layer_sequence[0]
                first_layer_keys = _get_layer_keys(self._seeker, first_layer_idx)
                self._next_future = self._executor.submit(
                    self._seeker.get_tensors, first_layer_keys, self.device, self.dtype
                )
                self._next_future_idx = first_layer_idx

        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
        **kwargs
    ):
        return cls(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            cache_to_ram=cache_to_ram,
            prefetch=prefetch,
        )
