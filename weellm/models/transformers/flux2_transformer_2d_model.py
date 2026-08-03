"""
transformer_streamer.py -- Hook-based layer-streaming for Flux2Transformer2DModel.

Pipeline strategy (3 stages, fully overlapped):
-------------------------------------------------
  Background thread  : live seek disk bytes -> GPU directly
  Main thread        : sync, apply weights, run forward pass

Timeline per block:
   Block N:   [ apply weights (fast) | -------- forward pass -------- ] [evict]
   Block N+1: [ ----------- GPU load in background thread ----------- ] [apply weights (fast)]
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.utils import clean_memory, report_memory
from weellm.live_seek import SafetensorsLiveSeeker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_layer_keys(seeker: SafetensorsLiveSeeker, prefix: str) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix + ".")]


def _get_resident_keys(seeker: SafetensorsLiveSeeker) -> List[str]:
    prefixes = ("transformer_blocks.", "single_transformer_blocks.")
    return [k for k in seeker.weight_map.keys() if not any(k.startswith(p) for p in prefixes)]


def _apply_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
):
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


# ---------------------------------------------------------------------------
# Flux2Transformer2DModelStreamer
# ---------------------------------------------------------------------------

class Flux2Transformer2DModelStreamer:
    """
    Wraps Flux2Transformer2DModel for memory-efficient layer streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker: SafetensorsLiveSeeker,
        double_block_count: int,
        single_block_count: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model = model
        self.seeker = seeker
        self.double_block_count = double_block_count
        self.single_block_count = single_block_count
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch

        # Ordered list: (shard_name, block_index, block_type)
        self._shard_order: List[tuple] = (
            [(f"transformer_blocks.{i}", i, "double") for i in range(double_block_count)]
            + [(f"single_transformer_blocks.{i}", i, "single") for i in range(single_block_count)]
        )
        self._shard_name_to_pos = {s: idx for idx, (s, _, _) in enumerate(self._shard_order)}

        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_future_name: Optional[str] = None
        self._lock = threading.Lock()

        self._install_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self):
        for shard_name, idx, btype in self._shard_order:
            block = (
                self.model.transformer_blocks[idx]
                if btype == "double"
                else self.model.single_transformer_blocks[idx]
            )
            block._flux_shard_name = shard_name
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    # ------------------------------------------------------------------
    # Pre-hook
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = module._flux_shard_name
        pos = self._shard_name_to_pos[shard_name]
        layer_keys = _get_layer_keys(self.seeker, shard_name)

        # ---- Apply current block weights --------------------------------
        with self._lock:
            if self.prefetch and self._next_future_name == shard_name and self._next_future is not None:
                sd = self._next_future.result()
                self._next_future = None
                self._next_future_name = None
            else:
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        _apply_state_dict(self.model, sd, self.device, self.dtype)
        module._flux_loaded_params = list(sd.keys())

        # ---- Launch background load for NEXT block ------------------
        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name = self._shard_order[next_pos][0]
            next_layer_keys = _get_layer_keys(self.seeker, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

    # ------------------------------------------------------------------
    # Post-hook
    # ------------------------------------------------------------------

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_flux_loaded_params", []))
        module._flux_loaded_params = []
        return output

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "Flux2Transformer2DModelStreamer":
        from diffusers import Flux2Transformer2DModel

        transformer_dir = Path(transformer_dir)

        print("=" * 60)
        print("Step 1/3 -- Initializing LiveSeeker on transformer weights ...")
        seeker = SafetensorsLiveSeeker(transformer_dir)
        print(f"Found {len(seeker.weight_map)} tensors across HF shards.")

        print("\nStep 2/3 -- Instantiating Flux2Transformer2DModel on meta device ...")
        with init_empty_weights():
            cfg = Flux2Transformer2DModel.load_config(str(transformer_dir / "config.json"))
            model = Flux2Transformer2DModel.from_config(cfg)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("\nStep 3/3 -- Loading resident tensors to GPU ...")
        resident_keys = _get_resident_keys(seeker)
        resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        double_count = len(model.transformer_blocks)
        single_count = len(model.single_transformer_blocks)
        print(f"\nInstalled {double_count} double blocks + {single_count} single blocks for streaming.")

        streamer = cls(
            model=model,
            seeker=seeker,
            double_block_count=double_count,
            single_block_count=single_count,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        print("\nFluxStreamer ready. Mode: Live Seek from original shards\n")
        return streamer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
