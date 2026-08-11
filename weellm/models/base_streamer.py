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
    ):
        self.model   = model
        self.seeker  = seeker
        self.device  = device
        self.dtype   = dtype
        self.prefetch = prefetch

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

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Write *state_dict* tensors into model parameters (meta → real device)."""
        place_tensors(self.model, state_dict, self.device, self.dtype)


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

        # Retrieve weights — either from a background prefetch future or synchronously.
        with self._lock:
            if (
                self.prefetch
                and self._next_future_name == shard_name
                and self._next_future is not None
            ):
                sd = self._next_future.result()
                self._next_future = None
                self._next_future_name = None
            else:
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        self.apply_state_dict(sd)
        del sd  # release CPU/GPU tensors immediately — the model now owns them

        # Clamp block INPUTS to prevent float16 overflow inside attention softmax.
        # IMPORTANT: pre_hook must *return* a tuple for PyTorch to replace the inputs;
        # assigning to a local variable and returning nothing is silently ignored.
        if self.dtype == torch.float16:
            clamped_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                    clamped_args.append(torch.clamp(arg, min=-60000.0, max=60000.0))
                else:
                    clamped_args.append(arg)
            args = tuple(clamped_args)

        # Launch background prefetch for the next block.
        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name, _ = self._shard_order[next_pos]
            next_keys    = self._get_layer_keys(next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

        # Return args so PyTorch replaces the module's inputs with the (possibly clamped) tuple.
        return args

    def _post_hook(self, module: nn.Module, args, output):
        evict_module(module)
        return output

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Shut down the background prefetch thread on garbage collection."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
