import importlib
import json
from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

def _apply_state_dict(model: nn.Module, state_dict: dict, device: str, dtype: torch.dtype):
    """Load weights to specific device and cast to dtype."""
    for name, tensor in state_dict.items():
        if tensor is not None:
            if "num_batches_tracked" in name:
                continue
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)

def _evict_params(model: nn.Module, param_names: List[str]):
    """Move named parameters back to the meta device (free VRAM)."""
    for name in param_names:
        set_module_tensor_to_device(model, name, "meta")

class LazyVAEStreamer:
    def __init__(self, model: nn.Module, seeker, device: str, dtype: torch.dtype):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        self._patched = False
        self._patch_decode()

    def _patch_decode(self):
        original_decode = self.model.decode

        def lazy_decode(self_obj, *args, **kwargs):
            print("\n[WeeLLM] Lazy VAE triggered! Pulling VAE weights directly to GPU...")
            
            keys = list(self.seeker.weight_map.keys())
            
            # 1. Load weights onto GPU
            state_dict = self.seeker.get_tensors(keys, self.device, self.dtype)
            _apply_state_dict(self.model, state_dict, self.device, self.dtype)
            del state_dict
            
            report_memory("After VAE Load")
            
            # 2. Execute actual decoding
            res = original_decode(*args, **kwargs)
            
            # 3. Evict weights to free VRAM for future loops
            print("\n[WeeLLM] Decoding complete. Evicting VAE weights back to meta device...")
            _evict_params(self.model, keys)
            clean_memory(self.device)
            
            report_memory("After VAE Eviction")
            
            return res

        self.model.decode = lazy_decode.__get__(self.model, self.model.__class__)
        self._patched = True

    @classmethod
    def from_pretrained(
        cls,
        vae_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False
    ) -> "LazyVAEStreamer":
        vae_dir = Path(vae_dir)
        
        # Determine VAE class name
        config_path = vae_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
            
        class_name = cfg_dict.get("_class_name", "AutoencoderKL")
        diffusers = importlib.import_module("diffusers")
        vae_cls = getattr(diffusers, class_name)
        
        print("\nStep 1/2 -- Initializing LiveSeeker on VAE weights ...")
        seeker = get_seeker(vae_dir, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors.")
        
        print(f"Step 2/2 -- Instantiating {class_name} on meta device ...")
        with init_empty_weights():
            cfg = vae_cls.load_config(str(config_path))
            model = vae_cls.from_config(cfg)
            
        model.eval()
        
        return cls(model, seeker, device, dtype)
