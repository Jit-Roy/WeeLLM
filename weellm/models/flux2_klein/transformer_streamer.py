"""
transformer_streamer.py -- Hook-based layer-streaming for Flux2Transformer2DModel.

Pipeline strategy (3 stages, fully overlapped):
-------------------------------------------------
  Background thread  : disk -> GPU directly via load_file(device='cuda')
  Main thread        : sync event, apply weights, run forward pass

Timeline per block:

   Block N:   [ apply weights (fast) | -------- forward pass -------- ] [evict]
   Block N+1: [ ----------- GPU load in background thread ----------- ] [apply weights (fast)]

Since per-block GPU compute (~398ms) >> direct GPU load (~80ms), the load
is completely hidden inside the forward pass => near-zero I/O overhead.

VRAM budget: resident (~390 MB) + current block (~490 MB)
             + next block being loaded in background (~490 MB)
             = ~2.3 GB peak (within the 4 GB budget)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file

from .splitter import split_flux_transformer, get_shard_path
from weellm.core.utils import clean_memory, report_memory


# ---------------------------------------------------------------------------
# Background CUDA loader
# ---------------------------------------------------------------------------

def _load_shard_to_cuda(
    path: str, device: str
) -> Tuple[Dict[str, torch.Tensor], torch.cuda.Event]:
    """
    Load a shard directly from disk to GPU in a background thread.

    safetensors.torch.load_file with device='cuda' uses CUDA DMA internally,
    completely bypassing the CPU pin_memory dance (~477ms saved per block).

    Returns the GPU tensor dict AND a CUDA event so the main thread can
    synchronise without a full device sync.
    """
    sd = load_file(path, device=device)
    event = torch.cuda.Event()
    event.record()          # Records on this thread's current CUDA stream
    return sd, event


def _load_shard_to_cpu(path: str) -> Dict[str, torch.Tensor]:
    """Fallback: load to CPU (used only on the cold first block)."""
    return load_file(path, device="cpu")


# ---------------------------------------------------------------------------
# Parameter management helpers
# ---------------------------------------------------------------------------

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
# FluxStreamer
# ---------------------------------------------------------------------------

class FluxStreamer:
    """
    Wraps Flux2Transformer2DModel for memory-efficient layer streaming.

    Every transformer block is stored as a separate shard on disk.
    A background thread pre-loads the NEXT block directly to GPU while the
    CURRENT block's forward pass runs, giving near-zero I/O overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        shard_dir: Path,
        double_block_count: int,
        single_block_count: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model = model
        self.shard_dir = shard_dir
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

        # Background GPU loader (1 worker -- sequential loads, no GPU contention)
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux_gpu_load") if prefetch else None
        )
        # Future holds (gpu_tensors, cuda_event) for the pre-loaded next block
        self._next_future = None
        self._next_future_name: Optional[str] = None

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
    # Pre-hook: apply current block weights + launch background load for next
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = module._flux_shard_name
        pos = self._shard_name_to_pos[shard_name]

        # ---- Apply current block weights --------------------------------
        if self.prefetch and self._next_future_name == shard_name and self._next_future is not None:
            # Block N's weights were pre-loaded by the background thread.
            # Get the result (instant -- load already finished during previous forward).
            gpu_sd, event = self._next_future.result()
            self._next_future = None
            self._next_future_name = None

            # Sync: ensure the background thread's CUDA ops are visible
            # to the main thread's stream before we use the tensors.
            torch.cuda.current_stream(self.device).wait_event(event)

            # Apply (tensors already on GPU -- set_module_tensor_to_device
            # with value already on target device is a fast pointer swap).
            _apply_state_dict(self.model, gpu_sd, self.device, self.dtype)
            module._flux_loaded_params = list(gpu_sd.keys())
        else:
            # Cold path: first block of each denoising step (no pre-load yet).
            # Load synchronously to CPU then transfer to GPU.
            path = str(get_shard_path(self.shard_dir, shard_name))
            cpu_sd = _load_shard_to_cpu(path)
            _apply_state_dict(self.model, cpu_sd, self.device, self.dtype)
            module._flux_loaded_params = list(cpu_sd.keys())

        # ---- Launch background GPU load for NEXT block ------------------
        if self.prefetch and self._executor is not None and pos + 1 < len(self._shard_order):
            next_name = self._shard_order[pos + 1][0]
            next_path = str(get_shard_path(self.shard_dir, next_name))
            self._next_future = self._executor.submit(
                _load_shard_to_cuda, next_path, self.device
            )
            self._next_future_name = next_name

    # ------------------------------------------------------------------
    # Post-hook: evict current block back to meta
    # ------------------------------------------------------------------

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_flux_loaded_params", []))
        module._flux_loaded_params = []
        # No empty_cache() -- allocator reuses freed blocks without explicit eviction.
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
        force_resplit: bool = False,
    ) -> "FluxStreamer":
        from diffusers import Flux2Transformer2DModel

        transformer_dir = Path(transformer_dir)

        print("=" * 60)
        print("Step 1/3 -- Splitting transformer into per-layer shards ...")
        shard_dir = split_flux_transformer(transformer_dir, force=force_resplit)

        print("\nStep 2/3 -- Instantiating Flux2Transformer2DModel on meta device ...")
        with init_empty_weights():
            cfg = Flux2Transformer2DModel.load_config(str(transformer_dir / "config.json"))
            model = Flux2Transformer2DModel.from_config(cfg)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("\nStep 3/3 -- Loading resident tensors to GPU ...")
        resident_sd = load_file(str(get_shard_path(shard_dir, "resident")), device="cpu")
        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        double_count = len(model.transformer_blocks)
        single_count = len(model.single_transformer_blocks)
        print(f"\nInstalled {double_count} double blocks + {single_count} single blocks for streaming.")

        streamer = cls(
            model=model,
            shard_dir=shard_dir,
            double_block_count=double_count,
            single_block_count=single_count,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        mode = "direct GPU load (disk->VRAM in background)" if prefetch else "sequential"
        print(f"\nFluxStreamer ready.  Mode: {mode}\n")
        return streamer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
