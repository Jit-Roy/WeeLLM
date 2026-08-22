"""
base_streamer.py -- Shared base class for all WeeLLM transformer/UNet streamers.

Every architecture-specific streamer inherits from ``BaseTransformerStreamer``
and only needs to implement:
  - ``_get_shard_order()``  -> ordered list of (shard_name, nn.Module) pairs
  - ``_get_resident_keys()`` -> list of weight-map keys that stay resident on GPU

The base class handles the full 3-stage pipeline to minimise GPU idle time:

  STAGE 1 — Disk -> CPU RAM  (background thread, N blocks ahead)
  STAGE 2 — CPU RAM -> VRAM  (non-blocking H2D DMA, done in pre_hook)
  STAGE 3 — VRAM -> GPU Compute (forward pass)

After Stage 2, the CPU RAM copy is freed immediately — RAM is a transit
buffer, not a persistent cache.  After Stage 3, the VRAM block is evicted
(unless it fits within the free VRAM budget, in which case it stays pinned
to eliminate disk I/O on future steps).

  Adaptive prefetch depth
  -----------------------
  At init the streamer measures available system RAM and the byte size of
  each block (from shard header metadata, no I/O). It then sets:

    prefetch_depth = min(MAX_PREFETCH, available_ram_budget // max_block_bytes)

  On a 4 GB RAM machine  → depth ~ 1-2   (minimal staging)
  On a 30 GB RAM machine → depth ~ 4+    (pipeline can absorb disk latency)

  Adaptive VRAM caching
  ---------------------
  After the first full inference pass, free VRAM is measured. Subsequent
  blocks are greedily pinned in VRAM (largest budget first) so later steps
  hit VRAM instead of disk → zero idle time for pinned blocks.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from weellm.memory import place_tensors, evict_module

logger = logging.getLogger("weellm")

# Attribute name stored on each streamed nn.Module block to track its shard name
_SHARD_NAME_ATTR = "_weellm_shard_name"

# Safety margin reserved for OS + PyTorch activations (not for staging blocks).
_RAM_SAFETY_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB
# Hard cap on prefetch depth — beyond this, benefits diminish rapidly.
_MAX_PREFETCH_DEPTH = 6
# Safety margin for VRAM budget (reserves headroom for activations during forward).
_VRAM_SAFETY_BYTES = 512 * 1024 * 1024        # 512 MB


class BaseTransformerStreamer(ABC):
    """
    Abstract base for all WeeLLM hook-based layer streamers.

    Subclasses MUST implement ``_get_shard_order()`` and
    ``_get_resident_keys()``. Everything else — pipeline scheduling, prefetch
    depth, VRAM calibration, executor management — is handled here.
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
        self.model  = model
        self.seeker = seeker
        self.device = device
        self.dtype  = dtype
        self.prefetch = prefetch
        # Where prefetched blocks land before H2D:
        #   "cpu"  → pinned CPU RAM  (safe on all machines)
        #   "cuda" → directly to VRAM (only if VRAM budget allows a 2nd block)
        self.prefetch_device: str = prefetch_device if prefetch_device else "cpu"

        # Build the ordered list of (shard_name, block_module) tuples.
        self._shard_order: List[Tuple[str, nn.Module]] = self._get_shard_order()
        self._shard_name_to_pos: Dict[str, int] = {
            name: idx for idx, (name, _) in enumerate(self._shard_order)
        }

        # ----------------------------------------------------------------
        # VRAM caching state (calibrated after the first full pass)
        # ----------------------------------------------------------------
        self._pinned_blocks: set = set()       # blocks permanently in VRAM
        self._cache_budget_bytes: int = 0      # remaining VRAM budget for pinning
        self._calibration_done: bool = False

        # ----------------------------------------------------------------
        # Adaptive prefetch depth
        # ----------------------------------------------------------------
        self._prefetch_depth: int = self._compute_prefetch_depth()

        # ----------------------------------------------------------------
        # Prefetch executor & future dict
        # ----------------------------------------------------------------
        # _prefetch_futures maps shard_name -> Future[state_dict] for
        # all blocks currently being read from disk in the background.
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=min(self._prefetch_depth, _MAX_PREFETCH_DEPTH),
                thread_name_prefix="weellm_prefetch",
            )
            if prefetch else None
        )
        self._prefetch_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

        self._install_hooks()

    # ------------------------------------------------------------------
    # Adaptive prefetch depth computation
    # ------------------------------------------------------------------

    def _compute_prefetch_depth(self) -> int:
        """
        Compute how many blocks to read ahead based on available RAM.
        Uses cached shard header metadata — no tensor loading.
        """
        if not self.prefetch or not self._shard_order:
            return 1

        # Compute block sizes from metadata (no disk I/O for actual tensors).
        try:
            block_bytes = {
                name: self.seeker.get_block_bytes(self._get_layer_keys(name))
                for name, _ in self._shard_order
            }
            max_block_bytes = max(block_bytes.values(), default=1)
        except Exception:
            logger.debug("[Streamer] Could not compute block sizes; defaulting prefetch_depth=1")
            return 1

        try:
            import psutil
            available = psutil.virtual_memory().available
            usable = max(0, available - _RAM_SAFETY_BYTES)
            depth = max(1, min(_MAX_PREFETCH_DEPTH, usable // max_block_bytes))
        except ImportError:
            depth = 1

        logger.info(
            "[Streamer] Adaptive prefetch depth: %d  "
            "(largest block: %.0f MB, available RAM: %.1f GB)",
            depth,
            max_block_bytes / 1e6,
            (available - _RAM_SAFETY_BYTES) / 1e9 if 'available' in dir() else 0,
        )
        return depth

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
        """Write *state_dict* tensors into model parameters (meta -> real device)."""
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
    # Pre-hook  (Stage 1 wait + Stage 2 H2D + next prefetch launch)
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = getattr(module, _SHARD_NAME_ATTR)
        pos = self._shard_name_to_pos[shard_name]
        layer_keys = self._get_layer_keys(shard_name)

        # VRAM calibration: reset peak stats at the very start of each pass.
        if not self._calibration_done and shard_name == self._shard_order[0][0]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

        is_pinned = shard_name in self._pinned_blocks

        if is_pinned:
            logger.debug(
                "    [Streamer] Block %s (%d/%d) [VRAM HIT — skipping disk]",
                shard_name, pos + 1, len(self._shard_order),
            )
        else:
            logger.debug(
                "    [Streamer] Block %s (%d/%d) [loading from %s]",
                shard_name, pos + 1, len(self._shard_order),
                "prefetch" if shard_name in self._prefetch_futures else "disk (sync)",
            )

        t0 = time.time()

        # ------------------------------------------------------------------
        # Stage 1: collect prefetch future (if ready) or sync-load from disk.
        # ------------------------------------------------------------------
        sd = None
        if not is_pinned:
            # Grab the future reference under the lock, release lock before .result().
            fut: Optional[Future] = None
            with self._lock:
                fut = self._prefetch_futures.pop(shard_name, None)

            if fut is not None:
                sd = fut.result()   # blocks until this specific block's read is done

            if sd is None:
                # Sync fallback: read directly to VRAM (skips unnecessary CPU staging).
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        t1 = time.time()

        # ------------------------------------------------------------------
        # Stage 2: H2D transfer (only if block came from CPU RAM).
        # Free the CPU copy immediately after — RAM is transit, not cache.
        # ------------------------------------------------------------------
        if not is_pinned:
            if self.prefetch and self.prefetch_device != self.device:
                sd = {k: v.to(self.device, non_blocking=True) for k, v in sd.items()}
                torch.cuda.synchronize()   # ensure DMA completes before forward()

            self.apply_state_dict(sd)
            torch.cuda.synchronize()       # flush CUDA copy_ ops from place_tensors
            del sd                          # ← CPU RAM freed here, immediately

        t2 = time.time()

        logger.debug(
            "    [Profile] %s (%d/%d): Disk/Wait=%.3fs | H2D+Apply=%.3fs",
            shard_name, pos + 1, len(self._shard_order), t1 - t0, t2 - t1,
        )
        setattr(module, "_weellm_t_compute_start", time.time())

        # ------------------------------------------------------------------
        # Clamp float16 inputs to prevent softmax overflow.
        # ------------------------------------------------------------------
        if self.dtype == torch.float16:
            args = tuple(
                torch.clamp(a, -60000.0, 60000.0)
                if isinstance(a, torch.Tensor) and a.is_floating_point()
                else a
                for a in args
            )

        # ------------------------------------------------------------------
        # Launch background prefetch for the next `prefetch_depth` blocks.
        # Done AFTER H2D so disk reads don't compete with the DMA bus.
        # Only submit futures for blocks not already queued and not VRAM-pinned.
        # ------------------------------------------------------------------
        if self.prefetch and self._executor is not None:
            for ahead in range(1, self._prefetch_depth + 1):
                npos = pos + ahead
                if npos >= len(self._shard_order):
                    break
                next_name, _ = self._shard_order[npos]
                # If already pinned in VRAM, no need to prefetch.
                if next_name in self._pinned_blocks:
                    continue
                with self._lock:
                    # Avoid submitting duplicate futures for the same block.
                    if next_name not in self._prefetch_futures:
                        nkeys = self._get_layer_keys(next_name)
                        pdev  = self.prefetch_device
                        self._prefetch_futures[next_name] = self._executor.submit(
                            self.seeker.get_tensors, nkeys, pdev, self.dtype
                        )

        return args

    # ------------------------------------------------------------------
    # Post-hook  (GPU compute timer + VRAM calibration + eviction)
    # ------------------------------------------------------------------

    def _post_hook(self, module: nn.Module, args, output):
        torch.cuda.synchronize()
        t_end  = time.time()
        t_start = getattr(module, "_weellm_t_compute_start", t_end)
        shard_name = getattr(module, _SHARD_NAME_ATTR)
        logger.debug("    [Profile] %s: GPU Compute=%.3fs", shard_name, t_end - t_start)

        should_evict = True

        # ------------------------------------------------------------------
        # VRAM calibration pass (first full pass only).
        # At the end of the first pass, measure peak VRAM reserved and derive
        # how many blocks we can permanently pin for free.
        # ------------------------------------------------------------------
        if not self._calibration_done:
            if shard_name == self._shard_order[-1][0] and torch.cuda.is_available():
                max_reserved = torch.cuda.max_memory_reserved(self.device)
                _, total_vram = torch.cuda.mem_get_info(self.device)
                self._cache_budget_bytes = max(
                    0, total_vram - max_reserved - _VRAM_SAFETY_BYTES
                )
                self._calibration_done = True
                logger.debug(
                    "    [Streamer] VRAM calibration done: peak_reserved=%.2f GB, "
                    "total=%.2f GB, pin_budget=%.2f GB",
                    max_reserved / 1e9, total_vram / 1e9,
                    self._cache_budget_bytes / 1e9,
                )

        # ------------------------------------------------------------------
        # VRAM pinning (after calibration).
        # Greedily keep blocks in VRAM if budget allows — eliminates disk I/O
        # for those blocks on all future inference steps.
        # ------------------------------------------------------------------
        else:
            if shard_name in self._pinned_blocks:
                should_evict = False
            elif self._cache_budget_bytes > 0:
                block_size = sum(
                    p.numel() * p.element_size()
                    for p in module.parameters()
                    if p.device.type != "meta"
                )
                if block_size > 0 and block_size <= self._cache_budget_bytes:
                    self._cache_budget_bytes -= block_size
                    self._pinned_blocks.add(shard_name)
                    should_evict = False
                    logger.debug(
                        "    [Streamer] Pinned %s to VRAM — remaining budget: %.2f GB",
                        shard_name, self._cache_budget_bytes / 1e9,
                    )

        if should_evict:
            evict_module(module)

        return output

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

        logger.info("  Instantiating %s on meta device ...", model_cls.__name__)
        with default_dtype(dtype), init_empty_weights():
            cfg   = model_cls.load_config(str(transformer_dir / "config.json"))
            model = model_cls.from_config(cfg)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        return model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Shut down the background prefetch executor on garbage collection."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
