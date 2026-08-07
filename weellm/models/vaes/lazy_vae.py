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
            mapped_name = _resolve_vae_key(model, name)
            set_module_tensor_to_device(model, mapped_name, device, value=tensor, dtype=dtype)

def _evict_params(model: nn.Module, param_names: List[str]):
    """Move named parameters back to the meta device (free VRAM)."""
    for name in param_names:
        mapped_name = _resolve_vae_key(model, name)
        set_module_tensor_to_device(model, mapped_name, "meta")


def _map_vae_key(name: str) -> str:
    mapped_name = name
    if ".query." in mapped_name:
        mapped_name = mapped_name.replace(".query.", ".to_q.")
    if ".key." in mapped_name:
        mapped_name = mapped_name.replace(".key.", ".to_k.")
    if ".value." in mapped_name:
        mapped_name = mapped_name.replace(".value.", ".to_v.")
    if ".proj_attn." in mapped_name:
        mapped_name = mapped_name.replace(".proj_attn.", ".to_out.0.")
    return mapped_name


def _resolve_vae_key(model: nn.Module, name: str) -> str:
    mapped_name = _map_vae_key(name)
    if _has_module_path(model, mapped_name):
        return mapped_name
    if _has_module_path(model, name):
        return name
    return mapped_name


def _has_module_path(model: nn.Module, name: str) -> bool:
    current = model
    parts = name.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
        if current is None:
            return False
    return hasattr(current, parts[-1])

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

            def _cast_tensor(value):
                if torch.is_tensor(value):
                    if value.device.type != self.device or (value.is_floating_point() and value.dtype != self.dtype):
                        return value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
                return value

            if args:
                first_arg = args[0]
                if torch.is_tensor(first_arg):
                    print(
                        f"[WeeLLM VAE Debug] decode input shape={tuple(first_arg.shape)} "
                        f"device={first_arg.device} dtype={first_arg.dtype}"
                    )
                    args = ( _cast_tensor(first_arg), ) + args[1:]
            if "z" in kwargs and torch.is_tensor(kwargs["z"]):
                z = kwargs["z"]
                print(
                    f"[WeeLLM VAE Debug] decode kwarg z shape={tuple(z.shape)} device={z.device} dtype={z.dtype}"
                )
                kwargs["z"] = _cast_tensor(z)
            
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
            
        # Flux2 VAE uses a BN layer that is accessed by the pipeline BEFORE decode() is called.
        # Eagerly load the bn layer buffers onto the CPU so they aren't meta tensors containing garbage data.
        if hasattr(model, "bn"):
            print("  Eagerly loading VAE BN layers to preserve contrast...")
            bn_keys = [k for k in seeker.weight_map.keys() if "bn." in k]
            bn_sd = seeker.get_tensors(bn_keys, device="cpu", dtype=torch.float32)
            _apply_state_dict(model, bn_sd, device="cpu", dtype=torch.float32)
            del bn_sd
            
        model.eval()
        
        return cls(model, seeker, device, dtype)
