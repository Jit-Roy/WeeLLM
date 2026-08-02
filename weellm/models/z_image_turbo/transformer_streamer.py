"""
transformer_streamer.py -- Hook-based layer-streaming for ZImageTransformer2DModel.

Pipeline strategy (3 stages, fully overlapped):
-------------------------------------------------
  GPU compute (layer N)
  Background thread: load shard N+1 disk -> CUDA
  CPU: scheduler / VAE / bookkeeping

The model is instantiated on the ``meta`` device (zero VRAM) and only the
currently-executing layer resides in VRAM.  Resident tensors (cap_embedder,
all_final_layer, all_x_embedder, cap_pad_token) are loaded once at startup.

Streaming layers (per forward hook pair):
  - layers.0  … layers.29          (30 main transformer layers)
  - context_refiner.0, .1          (2 context refiner layers)
  - noise_refiner.0, .1            (2 noise refiner layers)
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
from safetensors.torch import load_file

from .splitter import split_zimage_transformer, get_shard_path
from weellm.core.utils import clean_memory, report_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_shard(path: Path, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return load_file(str(path), device=device)


# ---------------------------------------------------------------------------
# ZImageStreamer
# ---------------------------------------------------------------------------

class ZImageStreamer:
    """
    Hook-based weight streamer for ZImageTransformer2DModel.

    Streams one layer at a time from pre-split per-layer shards on disk.
    """

    def __init__(
        self,
        model: nn.Module,
        shard_dir: Path,
        layer_names: List[str],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model       = model
        self.shard_dir   = shard_dir
        self.layer_names = layer_names   # ordered list of streamable layer names
        self.device      = device
        self.dtype       = dtype
        self.prefetch    = prefetch

        self._executor   = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_idx    = None
        self._lock        = threading.Lock()

        self._install_hooks()

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
        """Navigate dotted name to get a submodule from the model."""
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
        path = get_shard_path(self.shard_dir, name)

        # If the background thread already loaded this layer, use the result
        with self._lock:
            if self._next_idx == idx and self._next_future is not None:
                state = self._next_future.result()
                self._next_future = None
                self._next_idx    = None
            else:
                # synchronous fallback
                state = _load_shard(path, self.device, self.dtype)

        # Inject tensors into the module in-place
        for key, tensor in state.items():
            # Key format: "layers.N.sub.param" or "context_refiner.N.sub.param"
            # Strip the layer prefix to get the local parameter name
            local_key = ".".join(key.split(".")[2:])  # e.g. "attention.to_q.weight"
            try:
                # Must pass dtype= explicitly — set_module_tensor_to_device
                # ignores the dtype of 'value' and uses the model param's dtype
                # unless dtype= is given as a separate argument.
                set_module_tensor_to_device(
                    module, local_key, self.device, value=tensor, dtype=self.dtype
                )
            except Exception:
                pass  # some keys may be index-mapped differently; skip silently

        # Absolute safety net: cast every parameter in this module to target dtype.
        # Handles any edge cases where set_module_tensor_to_device ignored dtype.
        try:
            module.to(self.dtype)
        except Exception:
            pass

        # Prefetch next layer
        next_idx = idx + 1
        if self.prefetch and self._executor and next_idx < len(self.layer_names):
            next_name = self.layer_names[next_idx]
            next_path = get_shard_path(self.shard_dir, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    _load_shard, next_path, self.device, self.dtype
                )
                self._next_idx = next_idx

    def _post_hook(self, module: nn.Module, args, output):
        # Evict all parameters back to meta device to free VRAM
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
        force_resplit: bool = False,
    ) -> "ZImageStreamer":
        from diffusers import ZImageTransformer2DModel

        transformer_dir = Path(transformer_dir)

        # -----------------------------------------------------------
        # Step 1: Split into per-layer shards (once)
        # -----------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 1/3 -- Splitting ZImage transformer into per-layer shards ...")
        shard_dir = split_zimage_transformer(
            transformer_dir=transformer_dir,
            force=force_resplit,
        )

        # -----------------------------------------------------------
        # Step 2: Load model config, instantiate on meta device
        # -----------------------------------------------------------
        print("\nStep 2/3 -- Instantiating ZImageTransformer2DModel on meta device ...")
        config_path = transformer_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            import json
            config_dict = json.load(f)
        with init_empty_weights():
            transformer = ZImageTransformer2DModel.from_config(config_dict)

        # -----------------------------------------------------------
        # Step 3: Load resident tensors to GPU
        # -----------------------------------------------------------
        print("\nStep 3/3 -- Loading resident tensors to GPU ...")
        resident_shard = get_shard_path(shard_dir, "resident")
        resident_state = load_file(str(resident_shard), device=device)
        for key, tensor in resident_state.items():
            try:
                # NOTE: must pass dtype= explicitly; set_module_tensor_to_device
                # ignores the dtype of 'value' and uses the model param's original
                # dtype unless dtype= is specified as a separate argument.
                set_module_tensor_to_device(
                    transformer, key, device, value=tensor, dtype=dtype
                )
            except Exception:
                pass

        # Safety net: cast any remaining non-meta CUDA params to target dtype
        # (handles edge cases where set_module_tensor_to_device ignored dtype)
        def _cast_non_meta(module: nn.Module, tgt_dtype: torch.dtype) -> None:
            for name, param in list(module._parameters.items()):
                if (param is not None and not param.is_meta
                        and param.device.type != "meta"
                        and param.dtype != tgt_dtype):
                    module._parameters[name] = nn.Parameter(
                        param.data.to(tgt_dtype), requires_grad=False
                    )
            for child in module._modules.values():
                if child is not None:
                    _cast_non_meta(child, tgt_dtype)

        _cast_non_meta(transformer, dtype)

        report_memory("After resident load")

        # -----------------------------------------------------------
        # Step 4: Determine streaming layer order
        # -----------------------------------------------------------
        # Streaming layer order MUST match the actual forward() execution order:
        #   noise_refiner.0, noise_refiner.1   (run first: refine noisy x tokens)
        #   context_refiner.0, context_refiner.1 (run second: refine caption)
        #   layers.0 … layers.29                (main transformer blocks)
        layer_names = []
        n_layers = transformer.config.n_layers
        n_refiner = transformer.config.n_refiner_layers
        for i in range(n_refiner):
            layer_names.append(f"noise_refiner.{i}")
        for i in range(n_refiner):
            layer_names.append(f"context_refiner.{i}")
        for i in range(n_layers):
            layer_names.append(f"layers.{i}")

        print(f"\nInstalled {n_layers} main layers + {n_refiner} context + {n_refiner} noise refiner layers for streaming.")
        print("\nZImageStreamer ready.  Mode: direct GPU load (disk->VRAM in background)\n")

        return cls(
            model=transformer,
            shard_dir=shard_dir,
            layer_names=layer_names,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
