"""
base_te_streamer.py -- Abstract base for lazy-init decoder-only text encoder streamers.

All five decoder-style text encoder streamers (GLM, Gemma2, LLaMA, Qwen3, Mistral3)
share exactly the same:
  - __init__ boilerplate
  - _ensure_initialized() sequence
  - _place_tensors() / _evict_layer() helpers
  - _layer_pre_hook / _layer_post_hook / _capture_hook logic

Subclasses only need to override the handful of architecture-specific hooks below.

Pipeline design (mirrors BaseTransformerStreamer):
  STAGE 1 — Disk -> CPU RAM  (background thread, prefetch_depth layers ahead)
  STAGE 2 — CPU RAM -> VRAM  (non-blocking H2D, done synchronously in pre_hook)
  STAGE 3 — VRAM -> GPU Compute (forward pass)

CPU RAM copy is freed immediately after H2D. RAM is a transit buffer, not a cache.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from weellm.utils import default_dtype, clean_memory
from weellm.memory import place_tensors, evict_module
from weellm.seeker import get_seeker

logger = logging.getLogger("weellm")

# Safety margin reserved for OS + PyTorch activations
_RAM_SAFETY_BYTES  = 2 * 1024 * 1024 * 1024   # 2 GB
_MAX_PREFETCH_DEPTH = 4


class BaseLazyDecoderStreamer(ABC):
    """
    Abstract base for hook-based, lazy-initialised decoder-style text encoders.

    Subclasses MUST implement:
      - ``_model_name``          -> human-readable name used in log messages
      - ``_layer_prefix(idx)``  -> weight-map key prefix for layer *idx*
      - ``_resident_key_filter(key)`` -> True if *key* belongs to the resident set
      - ``_get_model_layers()`` -> the nn.ModuleList of transformer layers
      - ``_load_model_skeleton()`` -> instantiate self._model on meta device
      - ``_load_resident_extra()`` -> optional post-step after resident weights placed
      - ``_capture_layer_indices()`` -> set of layer indices whose outputs to capture

    Subclasses MAY override:
      - ``encode()``    -> default raises NotImplementedError
      - ``encode_ids()`` -> default raises NotImplementedError
      - ``__call__()``  -> default delegates to self._model
    """

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def _model_name(self) -> str:
        """Human-readable name, e.g. 'GLM' or 'Qwen3'."""

    @abstractmethod
    def _layer_prefix(self, idx: int) -> str:
        """Weight-map key prefix for transformer layer *idx*."""

    @abstractmethod
    def _resident_key_filter(self, key: str) -> bool:
        """Return True if *key* should stay resident (always loaded) on the GPU."""

    def _cpu_resident_key_filter(self, key: str) -> bool:
        """Return True if *key* should stay resident on the CPU. Default: False."""
        return False

    @abstractmethod
    def _get_model_layers(self) -> nn.ModuleList:
        """Return the nn.ModuleList of transformer layers from self._model."""

    @abstractmethod
    def _load_model_skeleton(self) -> None:
        """Instantiate self._model on meta device and set self._num_layers."""

    def _load_resident_extra(self) -> None:
        """Optional hook called after resident weights are placed. Default: no-op."""

    @abstractmethod
    def _capture_layer_indices(self) -> set:
        """Return the set of layer indices whose hidden-state outputs to capture."""

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        text_encoder_dir: Path | str,
        tokenizer_dir: Path | str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 512,
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.tokenizer_dir    = Path(tokenizer_dir)
        self.device           = device
        self.dtype            = dtype
        self.cache_to_ram     = cache_to_ram
        self.max_length       = max_length

        self._seeker       = None
        self._model: Optional[nn.Module] = None
        self._tokenizer    = None
        self._num_layers   = 0
        self._initialized  = False

        self._captured: Dict[int, torch.Tensor] = {}
        self._hook_handles: list = []

        # Prefetch executor and future dict (idx -> Future[state_dict])
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futures: Dict[int, Future] = {}
        self._prefetch_depth: int = 1   # updated after seeker is created
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Initialisation (lazy)                                               #
    # ------------------------------------------------------------------ #

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        logger.info("Initialising streaming %s text encoder ...", self._model_name)
        self._seeker = get_seeker(self.text_encoder_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_tokenizer()
        self._load_resident_modules()
        self._install_hooks()

        # Compute adaptive prefetch depth based on available RAM and layer size.
        self._prefetch_depth = self._compute_prefetch_depth()

        self._executor = ThreadPoolExecutor(
            max_workers=min(self._prefetch_depth, _MAX_PREFETCH_DEPTH),
            thread_name_prefix="te_prefetch",
        )
        self._initialized = True
        logger.info("Text encoder ready (streaming via Live Seek).")

    def _compute_prefetch_depth(self) -> int:
        """Adaptive prefetch depth: available_ram_budget // avg_layer_size."""
        if self._num_layers == 0:
            return 1
        try:
            # Sample layer 0 size as representative for all layers.
            sample_keys = [
                k for k in self._seeker.weight_map
                if k.startswith(self._layer_prefix(0))
            ]
            layer_bytes = self._seeker.get_block_bytes(sample_keys)
            import psutil
            available = psutil.virtual_memory().available
            usable = max(0, available - _RAM_SAFETY_BYTES)
            depth = max(1, min(_MAX_PREFETCH_DEPTH, usable // layer_bytes))
            logger.debug(
                "[TE Streamer] Adaptive prefetch depth: %d "
                "(layer=%.0f MB, usable RAM=%.1f GB)",
                depth, layer_bytes / 1e6, usable / 1e9,
            )
            return depth
        except Exception:
            return 1

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_dir), trust_remote_code=False
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_dir), trust_remote_code=True
            )

    def _load_resident_modules(self) -> None:
        gpu_keys = [k for k in self._seeker.weight_map if self._resident_key_filter(k)]
        cpu_keys = [k for k in self._seeker.weight_map if self._cpu_resident_key_filter(k)]

        if cpu_keys:
            cpu_sd = self._seeker.get_tensors(cpu_keys, device="cpu", dtype=self.dtype)
            place_tensors(self._model, cpu_sd, "cpu", self.dtype)
            del cpu_sd

        if gpu_keys:
            gpu_sd = self._seeker.get_tensors(gpu_keys, device=self.device, dtype=self.dtype)
            self._place_tensors(gpu_sd)
            del gpu_sd

        self._load_resident_extra()
        clean_memory(self.device)

    # ------------------------------------------------------------------ #
    # Tensor helpers                                                       #
    # ------------------------------------------------------------------ #

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor]) -> None:
        place_tensors(self._model, state_dict, self.device, self.dtype)

    # ------------------------------------------------------------------ #
    # Hook installation                                                    #
    # ------------------------------------------------------------------ #

    def _install_hooks(self) -> None:
        capture_set = self._capture_layer_indices()
        for layer_idx in range(self._num_layers):
            layer = self._get_model_layers()[layer_idx]
            layer._te_layer_idx = layer_idx
            h_pre  = layer.register_forward_pre_hook(self._layer_pre_hook)
            h_post = layer.register_forward_hook(self._layer_post_hook)
            self._hook_handles.extend([h_pre, h_post])
            if layer_idx in capture_set:
                h_cap = layer.register_forward_hook(self._capture_hook)
                self._hook_handles.append(h_cap)

    # ------------------------------------------------------------------ #
    # Hooks                                                                #
    # ------------------------------------------------------------------ #

    def _layer_pre_hook(self, module: nn.Module, args):
        idx        = module._te_layer_idx
        layer_keys = [k for k in self._seeker.weight_map if k.startswith(self._layer_prefix(idx))]

        t0 = time.time()

        # ----------------------------------------------------------------
        # Stage 1: collect prefetch future or sync-load from disk.
        # Grab the future reference under the lock, then release before .result().
        # ----------------------------------------------------------------
        fut: Optional[Future] = None
        with self._lock:
            fut = self._prefetch_futures.pop(idx, None)

        sd = fut.result() if fut is not None else None

        if sd is None:
            # Sync fallback: load directly to CPU (then H2D below).
            sd = self._seeker.get_tensors(layer_keys, device="cpu", dtype=self.dtype)

        t1 = time.time()

        # ----------------------------------------------------------------
        # Stage 2: H2D transfer (non-blocking DMA), then free CPU copy.
        # ----------------------------------------------------------------
        sd = {k: v.to(self.device, non_blocking=True) for k, v in sd.items()}
        torch.cuda.synchronize()   # ensure DMA is complete before forward()

        self._place_tensors(sd)
        torch.cuda.synchronize()   # flush CUDA copy_ ops from place_tensors
        del sd                      # CPU RAM freed immediately after H2D

        t2 = time.time()
        logger.debug(
            "    [TE Profile] layer %d (%d/%d): Disk/Wait=%.3fs | H2D=%.3fs",
            idx, idx + 1, self._num_layers, t1 - t0, t2 - t1,
        )

        # ----------------------------------------------------------------
        # Launch background prefetch for the next `prefetch_depth` layers.
        # Done AFTER H2D so disk reads don't compete with the DMA bus.
        # ----------------------------------------------------------------
        if self._executor is not None:
            for ahead in range(1, self._prefetch_depth + 1):
                nidx = idx + ahead
                if nidx >= self._num_layers:
                    break
                with self._lock:
                    if nidx not in self._prefetch_futures:
                        nkeys = [
                            k for k in self._seeker.weight_map
                            if k.startswith(self._layer_prefix(nidx))
                        ]
                        # Prefetch to CPU pinned RAM (safe on all hardware sizes).
                        self._prefetch_futures[nidx] = self._executor.submit(
                            self._seeker.get_tensors, nkeys, "cpu", self.dtype
                        )

    def _layer_post_hook(self, module: nn.Module, args, output):
        evict_module(module)
        return output

    def _capture_hook(self, module: nn.Module, args, output):
        idx    = module._te_layer_idx
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured[idx] = hidden.detach().clone()
        return output

    # ------------------------------------------------------------------ #
    # Encoding — override in subclasses                                   #
    # ------------------------------------------------------------------ #

    def encode(self, prompt) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement encode()")

    def encode_ids(self, input_ids, attention_mask=None) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement encode_ids()")

    # ------------------------------------------------------------------ #
    # Properties / lifecycle                                              #
    # ------------------------------------------------------------------ #

    @property
    def tokenizer(self):
        self._ensure_initialized()
        return self._tokenizer

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __del__(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
