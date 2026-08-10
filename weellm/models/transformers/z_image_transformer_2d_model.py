"""
z_image_transformer_2d_model.py -- Hook-based layer-streaming for ZImageTransformer2DModel.

Pipeline strategy (3 stages, fully overlapped):
-------------------------------------------------
  GPU compute (layer N)
  Background thread: live seek bytes for N+1 disk -> CUDA
  CPU: scheduler / VAE / bookkeeping

The model is instantiated on the ``meta`` device (zero VRAM) and only the
currently-executing layer resides in VRAM. Resident tensors are loaded once at startup.

ZImageTransformer has an unusual architecture: noise_refiner, context_refiner,
and main layers are all named by string paths (e.g. ``"noise_refiner.0"``,
``"layers.3"``), so this streamer uses the named-path hook pattern rather than
inheriting BaseTransformerStreamer (which assumes indexed block lists).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.utils import clean_memory, report_memory
from weellm.seeker import get_seeker

logger = logging.getLogger("weellm")


def _get_layer_keys(seeker, prefix: str) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix + ".")]


def _get_resident_keys(seeker) -> List[str]:
    prefixes = ("layers.", "context_refiner.", "noise_refiner.")
    return [k for k in seeker.weight_map.keys() if not any(k.startswith(p) for p in prefixes)]


class ZImageTransformer2DModelStreamer:
    """
    Hook-based weight streamer for ZImageTransformer2DModel.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        layer_names: List[str],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model       = model
        self.seeker      = seeker
        self.layer_names = layer_names
        self.device      = device
        self.dtype       = dtype
        self.prefetch    = prefetch

        self._executor    = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_idx    = None
        self._lock        = threading.Lock()

        self._install_hooks()

    def __del__(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        for idx, name in enumerate(self.layer_names):
            module = self._get_submodule(name)
            module._wee_layer_name = name
            module._wee_layer_idx  = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    def _get_submodule(self, name: str) -> nn.Module:
        """Navigate a dotted name (e.g. ``"layers.3"``) to get a submodule."""
        parts = name.split(".")
        m = self.model
        for p in parts:
            m = getattr(m, p) if not p.isdigit() else m[int(p)]
        return m

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        idx  = module._wee_layer_idx
        name = module._wee_layer_name
        layer_keys = _get_layer_keys(self.seeker, name)

        with self._lock:
            if self._next_idx == idx and self._next_future is not None:
                state = self._next_future.result()
                self._next_future = None
                self._next_idx    = None
            else:
                state = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        for key, tensor in state.items():
            local_key = ".".join(key.split(".")[2:])
            try:
                set_module_tensor_to_device(
                    module, local_key, self.device, value=tensor, dtype=self.dtype
                )
            except Exception:
                pass

        try:
            module.to(self.dtype)
        except Exception:
            pass

        next_idx = idx + 1
        if self.prefetch and self._executor and next_idx < len(self.layer_names):
            next_name      = self.layer_names[next_idx]
            next_layer_keys = _get_layer_keys(self.seeker, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_idx = next_idx

    def _post_hook(self, module: nn.Module, args, output):
        for name, param in list(module.named_parameters(recurse=True)):
            try:
                set_module_tensor_to_device(module, name, "meta")
            except Exception:
                pass
        clean_memory(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        **kwargs,
    ) -> "ZImageTransformer2DModelStreamer":
        from diffusers import ZImageTransformer2DModel

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("Step 2/3 -- Instantiating ZImageTransformer2DModel on meta device ...")
        config_path = transformer_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        with init_empty_weights():
            transformer = ZImageTransformer2DModel.from_config(config_dict)

        logger.info("Step 3/3 -- Loading resident tensors to GPU ...")
        resident_keys  = _get_resident_keys(seeker)
        resident_state = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for key, tensor in resident_state.items():
            try:
                set_module_tensor_to_device(transformer, key, device, value=tensor, dtype=dtype)
            except Exception:
                pass

        # Safety: ensure no resident tensors are wrong dtype
        def _cast_non_meta(module: nn.Module, tgt_dtype: torch.dtype) -> None:
            for name, param in list(module._parameters.items()):
                if (
                    param is not None
                    and not param.is_meta
                    and param.device.type != "meta"
                    and param.dtype != tgt_dtype
                ):
                    module._parameters[name] = nn.Parameter(
                        param.data.to(tgt_dtype), requires_grad=False
                    )
            for child in module._modules.values():
                if child is not None:
                    _cast_non_meta(child, tgt_dtype)

        _cast_non_meta(transformer, dtype)
        report_memory("After resident load")

        n_layers  = transformer.config.n_layers
        n_refiner = transformer.config.n_refiner_layers
        layer_names: List[str] = []
        for i in range(n_refiner):
            layer_names.append(f"noise_refiner.{i}")
        for i in range(n_refiner):
            layer_names.append(f"context_refiner.{i}")
        for i in range(n_layers):
            layer_names.append(f"layers.{i}")

        logger.info(
            "Installed %d main layers + %d context + %d noise refiner layers for streaming.",
            n_layers, n_refiner, n_refiner,
        )
        logger.info("ZImageStreamer ready. Mode: Live Seek from original shards")

        return cls(
            model=transformer,
            seeker=seeker,
            layer_names=layer_names,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
