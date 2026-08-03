"""
transformer_streamer.py -- Hook-based layer-streaming for SD3Transformer2DModel.

Architecture (SD3.5 Medium):
  - 24 joint transformer blocks (transformer_blocks.0..23)
  - No single_transformer_blocks (unlike FLUX.1)

Strategy:
  - Resident on GPU: context_embedder, pos_embed, time_text_embed,
                     norm_out, proj_out  (small, always needed)
  - Streamed: transformer_blocks[i] (loaded just-in-time, evicted after)

Single safetensors file: transformer/diffusion_pytorch_model.safetensors
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.utils import clean_memory, report_memory
from weellm.seeker import get_seeker


def _get_layer_keys(seeker, prefix: str) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix + ".")]


def _get_resident_keys(seeker) -> List[str]:
    streaming_prefixes = ("transformer_blocks.",)
    return [k for k in seeker.weight_map.keys() if not any(k.startswith(p) for p in streaming_prefixes)]


def _apply_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
    """Write tensors into model parameters (handles meta -> real device)."""
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)
        else:
            set_module_tensor_to_device(model, name, device, value=tensor)


def _evict_params(model: nn.Module, param_names: List[str]):
    """Move named parameters back to the meta device (free VRAM)."""
    for name in param_names:
        set_module_tensor_to_device(model, name, "meta")


class SD3Transformer2DModelStreamer:
    """
    Wraps SD3Transformer2DModel (SD 3.5 Medium) for memory-efficient layer streaming.
    Streams directly from the original Hugging Face safetensors shard via live seek.

    Architecture: 24 joint transformer blocks, no single-stream blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        block_count: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model = model
        self.seeker = seeker
        self.block_count = block_count
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch

        self._shard_order = [f"transformer_blocks.{i}" for i in range(block_count)]
        self._shard_name_to_pos = {s: idx for idx, s in enumerate(self._shard_order)}

        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_future_name: Optional[str] = None
        self._lock = threading.Lock()

        self._install_hooks()

    def _install_hooks(self):
        for i, shard_name in enumerate(self._shard_order):
            block = self.model.transformer_blocks[i]
            block._sd35_shard_name = shard_name
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = module._sd35_shard_name
        pos = self._shard_name_to_pos[shard_name]
        layer_keys = _get_layer_keys(self.seeker, shard_name)

        with self._lock:
            if self.prefetch and self._next_future_name == shard_name and self._next_future is not None:
                sd = self._next_future.result()
                self._next_future = None
                self._next_future_name = None
            else:
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        _apply_state_dict(self.model, sd, self.device, self.dtype)
        module._sd35_loaded_params = list(sd.keys())

        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name = self._shard_order[next_pos]
            next_layer_keys = _get_layer_keys(self.seeker, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_sd35_loaded_params", []))
        module._sd35_loaded_params = []
        return output

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False
    ) -> "SD3Transformer2DModelStreamer":
        from diffusers import SD3Transformer2DModel

        transformer_dir = Path(transformer_dir)

        print("Step 1/3 -- Initializing LiveSeeker on SD3.5 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors.")

        print("\nStep 2/3 -- Instantiating SD3Transformer2DModel on meta device ...")
        with init_empty_weights():
            cfg = SD3Transformer2DModel.load_config(str(transformer_dir / "config.json"))
            model = SD3Transformer2DModel.from_config(cfg)
        model.eval()

        # Move non-meta buffers to device (e.g., position embeddings)
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("\nStep 3/3 -- Loading resident SD3.5 transformer tensors to GPU ...")
        resident_keys = _get_resident_keys(seeker)
        resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        block_count = len(model.transformer_blocks)
        print(f"\nInstalled {block_count} joint transformer blocks for streaming.")

        streamer = cls(
            model=model,
            seeker=seeker,
            block_count=block_count,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        print("SD3Transformer2DModelStreamer ready. Mode: Live Seek from original shard\n")
        return streamer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
