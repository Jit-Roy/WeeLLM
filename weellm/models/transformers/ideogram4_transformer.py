"""
ideogram4_transformer.py -- Hook-based layer-streaming for Ideogram 4 diffusion transformer.

Uses the Double-Stream buffer overlap to load the massive 34 FP8 blocks directly from the SSD
while maintaining under 4.0 GB VRAM footprint.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.seeker import get_seeker
from weellm.utils import clean_memory


def _get_layer_keys(seeker, layer_idx: int) -> List[str]:
    prefix = f"layers.{layer_idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]

def _get_resident_keys(seeker) -> List[str]:
    keys = []
    for k in seeker.weight_map.keys():
        if not k.startswith("layers."):
            keys.append(k)
    return keys


class Ideogram4Transformer2DModelStreamer:
    def __init__(
        self,
        model_dir: Path | str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.cache_to_ram = cache_to_ram

        self._seeker = None
        self._model = None
        self._num_layers = 0
        self._initialized = False

        self._executor = None
        self._next_future = None
        self._next_future_idx = None
        self._lock = threading.Lock()
        
        self._ensure_initialized()

    def _ensure_initialized(self):
        if self._initialized:
            return
        print("Initialising streaming Ideogram4 transformer ...")
        self._seeker = get_seeker(self.model_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_resident_modules()
        self._install_hooks()
        
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ideo_gpu_load"
        )
        self._initialized = True
        print("Ideogram 4 transformer ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        # We import here so it doesn't break if diffusers is old
        import diffusers
        if not hasattr(diffusers, "Ideogram4Transformer2DModel"):
            raise ImportError("Diffusers does not have Ideogram4Transformer2DModel. Update diffusers.")
        
        config = diffusers.Ideogram4Transformer2DModel.load_config(str(self.model_dir))
        with init_empty_weights():
            self._model = diffusers.Ideogram4Transformer2DModel.from_config(config)
        self._model.eval()
        
        # Monkey-patch .to() to prevent diffusers from crashing on meta tensors
        def _noop_to(self, *args, **kwargs):
            return self
        self._model.__class__.to = _noop_to
        
        # Determine number of layers
        if hasattr(self._model, "layers"):
            self._num_layers = len(self._model.layers)
        elif hasattr(self._model, "transformer_blocks"):
            self._num_layers = len(self._model.transformer_blocks)
        else:
            self._num_layers = 34

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        self._place_tensors(resident_sd)
        del resident_sd
        
        # Move all registered buffers to GPU (like inv_freq) since .to() is patched
        for name, buf in list(self._model.named_buffers()):
            if buf.device.type != self.device:
                set_module_tensor_to_device(self._model, name, self.device, value=buf.to(self.device, dtype=self.dtype if buf.is_floating_point() else None))

        clean_memory(self.device)

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor]):
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
                    self._model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self._model, name, self.device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            if name.endswith(".weight_scale"):
                continue
            set_module_tensor_to_device(self._model, name, "meta")

    def _install_hooks(self):
        layers = getattr(self._model, "layers", getattr(self._model, "transformer_blocks", None))
        if layers is None:
            raise ValueError("Could not find layers in Ideogram4Transformer2DModel.")
            
        for layer_idx in range(self._num_layers):
            layer = layers[layer_idx]
            layer._ideo_layer_idx = layer_idx

            layer.register_forward_pre_hook(self._layer_pre_hook)
            layer.register_forward_hook(self._layer_post_hook)

    def _map_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mapped = {}
        for k, v in state_dict.items():
            if ".attention.o." in k:
                mapped[k.replace(".attention.o.", ".attention.to_out.0.")] = v
            elif ".attention.qkv." in k:
                q, k_val, v_val = v.chunk(3, dim=0)
                prefix = k.split(".attention.qkv.")[0] + ".attention."
                suffix = k.split(".attention.qkv.")[1]
                mapped[f"{prefix}to_q.{suffix}"] = q
                mapped[f"{prefix}to_k.{suffix}"] = k_val
                mapped[f"{prefix}to_v.{suffix}"] = v_val
            else:
                mapped[k] = v
        return mapped

    def _layer_pre_hook(self, module: nn.Module, args):
        idx = module._ideo_layer_idx
        layer_keys = _get_layer_keys(self._seeker, idx)

        with self._lock:
            if self._next_future_idx == idx and self._next_future is not None:
                gpu_sd = self._next_future.result()
                self._next_future = None
                self._next_future_idx = None
            else:
                gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        mapped_sd = self._map_state_dict(gpu_sd)
        self._place_tensors(mapped_sd)
        module._ideo_loaded_sd = mapped_sd

        next_idx = idx + 1
        if next_idx < self._num_layers and self._executor is not None:
            next_layer_keys = _get_layer_keys(self._seeker, next_idx)
            with self._lock:
                self._next_future = self._executor.submit(
                    self._seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_future_idx = next_idx

    def _layer_post_hook(self, module: nn.Module, args, output):
        self._evict_layer(getattr(module, "_ideo_loaded_sd", {}))
        module._ideo_loaded_sd = {}
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        **kwargs,
    ) -> "Ideogram4Transformer2DModelStreamer":
        return cls(
            model_dir=model_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
        )
