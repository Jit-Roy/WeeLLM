"""
base_streamer.py -- Shared base class for all WeeLLM transformer/UNet streamers.

Every architecture-specific streamer inherits from ``BaseTransformerStreamer``
and only needs to implement:
  - ``_get_shard_order()``  → ordered list of (shard_name, nn.Module) pairs
  - ``_get_resident_keys()`` → list of weight-map keys that stay resident on GPU

The base class handles:
  - apply_state_dict / evict_params (meta-device eviction)
  - Generic pre_hook / post_hook with background prefetching
  - ThreadPoolExecutor lifecycle (creation + __del__ shutdown)
"""

from __future__ import annotations

import logging
import threading
import gc
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from weellm.memory import place_tensors, evict_module

logger = logging.getLogger("weellm")

# Attribute name stored on each streamed nn.Module block to track its shard name
_SHARD_NAME_ATTR = "_weellm_shard_name"


class BaseTransformerStreamer(ABC):
    """
    Abstract base for all WeeLLM hook-based layer streamers.

    Subclasses MUST implement ``_get_shard_order()`` and
    ``_get_resident_keys()``. Everything else — hooks, prefetching, executor
    management — is handled here.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        prefetch_device: Optional[str] = None,
    ):
        self.model   = model
        self.seeker  = seeker
        self.device  = device
        self.dtype   = dtype
        self.prefetch = prefetch
        # Where to load prefetched blocks:
        #   None / "cuda" → directly to VRAM (may OOM on 4 GB cards)
        #   "cpu"         → pinned CPU RAM; moved to VRAM in _pre_hook (safe)
        self.prefetch_device: str = prefetch_device if prefetch_device else "cpu"

        # Build the ordered list of (shard_name, block_module) tuples.
        self._shard_order: List[Tuple[str, nn.Module]] = self._get_shard_order()
        self._shard_name_to_pos: Dict[str, int] = {
            name: idx for idx, (name, _) in enumerate(self._shard_order)
        }

        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="weellm_gpu_load")
            if prefetch else None
        )
        self._next_future = None
        self._next_future_name: Optional[str] = None
        self._lock = threading.Lock()

        self._install_hooks()

    # ------------------------------------------------------------------
    # Abstract interface — implement in each architecture subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        """
        Return an ordered list of ``(shard_prefix, block_module)`` pairs.

        The order determines the sequence in which blocks are pre-fetched and
        streamed. Example for a Flux-style architecture::

            return [
                (f"transformer_blocks.{i}", self.model.transformer_blocks[i])
                for i in range(self.double_block_count)
            ] + [
                (f"single_transformer_blocks.{i}", self.model.single_transformer_blocks[i])
                for i in range(self.single_block_count)
            ]
        """

    @abstractmethod
    def _get_resident_keys(self) -> List[str]:
        """
        Return the list of weight-map keys that should be loaded once at
        startup and kept resident on the GPU throughout inference.

        Example::

            streaming = ("transformer_blocks.", "single_transformer_blocks.")
            return [k for k in self.seeker.weight_map if not any(k.startswith(p) for p in streaming)]
        """

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        """Write *state_dict* tensors into model parameters (meta → real device)."""
        place_tensors(self.model, state_dict, self.device, self.dtype, skip_errors=skip_errors)


    def _get_layer_keys(self, shard_name: str) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if k.startswith(shard_name + ".")
        ]

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        for shard_name, block in self._shard_order:
            setattr(block, _SHARD_NAME_ATTR, shard_name)
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = getattr(module, _SHARD_NAME_ATTR)
        pos = self._shard_name_to_pos[shard_name]
        layer_keys = self._get_layer_keys(shard_name)
        
        if not getattr(self, "_calibration_done", False):
            if not hasattr(self, "_pinned_blocks"):
                self._pinned_blocks = set()
                self._cache_budget_bytes = 0
            if shard_name == self._shard_order[0][0] and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        
        is_pinned = getattr(self, "_pinned_blocks", None) is not None and shard_name in self._pinned_blocks
        
        # Add a visual progress log
        if is_pinned:
            logger.debug("    [Streamer] Streaming block %s (%d/%d) [VRAM CACHED]", shard_name, pos+1, len(self._shard_order))
        else:
            logger.debug("    [Streamer] Streaming block %s (%d/%d) ...", shard_name, pos+1, len(self._shard_order))

        import time
        t0 = time.time()

        sd = None
        # Retrieve weights — either from a background prefetch future or synchronously.
        # FIX #3: Grab the future reference under the lock, then release the lock
        # before calling .result(). This prevents the prefetch thread from being
        # blocked by the lock while we wait for its result.
        fut = None
        with self._lock:
            if (
                self.prefetch
                and self._next_future_name == shard_name
                and self._next_future is not None
            ):
                fut = self._next_future
                self._next_future = None
                self._next_future_name = None

        # Wait for prefetch result OUTSIDE the lock.
        if fut is not None:
            sd = fut.result()

        if not is_pinned:
            if sd is None:
                # Sync fallback: load directly to the target device (CUDA).
                # Do NOT use prefetch_device here — that forces an unnecessary CPU
                # staging step which competes with H2D DMA on the memory bus.
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

            # H2D transfer: move pinned CPU tensors to VRAM (non-blocking DMA).
            if self.prefetch and self.prefetch_device != self.device:
                sd = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in sd.items()
                }
                torch.cuda.synchronize()  # ensure H2D DMA is complete before forward

        t1 = time.time()

        if not is_pinned:
            self.apply_state_dict(sd)
            # Synchronize after apply_state_dict: the set_module_tensor_to_device calls
            # inside place_tensors issue CUDA copy_ ops. We must flush them before forward()
            # starts, otherwise their cost leaks invisibly into the GPU compute timer.
            torch.cuda.synchronize()
            del sd  # release CPU tensors immediately

        t2 = time.time()

        logger.debug("    [Profile] %s (%d/%d): Disk/Wait=%.3fs | H2D Transfer=%.3fs", shard_name, pos+1, len(self._shard_order), t1-t0, t2-t1)

        setattr(module, "_weellm_compute_start", time.time())

        # Clamp block INPUTS to prevent float16 overflow inside attention softmax.
        if self.dtype == torch.float16:
            clamped_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                    clamped_args.append(torch.clamp(arg, min=-60000.0, max=60000.0))
                else:
                    clamped_args.append(arg)
            args = tuple(clamped_args)

        # Launch background prefetch for the NEXT block AFTER all current-block work
        # (H2D + apply_state_dict + sync) is complete. This ensures the background
        # RAM read thread does not compete with the foreground DMA on the memory bus.
        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name, _ = self._shard_order[next_pos]
            is_next_pinned = (
                getattr(self, "_pinned_blocks", None) is not None
                and next_name in self._pinned_blocks
            )
            if not is_next_pinned:
                next_keys = self._get_layer_keys(next_name)
                _pdev = self.prefetch_device
                with self._lock:
                    self._next_future = self._executor.submit(
                        self.seeker.get_tensors, next_keys, _pdev, self.dtype
                    )
                    self._next_future_name = next_name

        return args

    def _post_hook(self, module: nn.Module, args, output):
        torch.cuda.synchronize()
        import time
        t_end = time.time()
        t_start = getattr(module, "_weellm_compute_start", t_end)
        shard_name = getattr(module, _SHARD_NAME_ATTR)
        pos = self._shard_name_to_pos[shard_name]
        logger.debug("    [Profile] %s: GPU Compute=%.3fs", shard_name, t_end-t_start)

        should_evict = True

        if not getattr(self, "_calibration_done", False):
            if shard_name == self._shard_order[-1][0]:
                if torch.cuda.is_available():
                    max_reserved = torch.cuda.max_memory_reserved(self.device)
                    free, total = torch.cuda.mem_get_info(self.device)
                    # Use max_reserved to account for PyTorch's allocator fragmentation overhead
                    self._cache_budget_bytes = total - max_reserved - (512 * 1024 * 1024)
                    self._calibration_done = True
                    logger.debug("    [Streamer] VRAM Calibration: Peak Reserved=%.2fGB, Total=%.2fGB, Safe Budget for Caching=%.2fGB", max_reserved/1024**3, total/1024**3, max(0, self._cache_budget_bytes)/1024**3)
        else:
            if shard_name in self._pinned_blocks:
                should_evict = False
            elif getattr(self, "_cache_budget_bytes", 0) > 0:
                # Can we fit this block in VRAM?
                block_size = sum(
                    p.numel() * p.element_size()
                    for p in module.parameters()
                    if getattr(p, "device", None) is not None and p.device.type != "meta"
                )
                if block_size > 0 and block_size <= self._cache_budget_bytes:
                    self._cache_budget_bytes -= block_size
                    self._pinned_blocks.add(shard_name)
                    should_evict = False
                    logger.debug("    [Streamer] Pinned %s to VRAM. Remaining VRAM budget: %.2fGB", shard_name, self._cache_budget_bytes/1e9)

        if should_evict:
            evict_module(module)

        return output

        # (Removed __call__ as it is bypassed by diffusers pipeline)

    # ------------------------------------------------------------------
    # Common from_pretrained helper
    # ------------------------------------------------------------------

    @classmethod
    def _load_model_on_meta(cls, model_cls, transformer_dir, device, dtype, seeker):
        """
        Shared 3-step init:
          1. Instantiate model on meta device.
          2. Move non-meta buffers to device.
          3. Load resident keys to GPU.

        Returns the ready model.
        """
        from accelerate import init_empty_weights
        from weellm.utils import default_dtype
        from weellm.utils import clean_memory, report_memory

        logger.info("  Instantiating %s on meta device ...", model_cls.__name__)
        with default_dtype(dtype), init_empty_weights():
            cfg   = model_cls.load_config(str(transformer_dir / "config.json"))
            model = model_cls.from_config(cfg)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        return model
