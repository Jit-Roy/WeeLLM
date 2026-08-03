"""
transformer_streamer.py -- Hook-based layer-streaming for QwenImageTransformer2DModel.

Architecture (Qwen-Image):
  - 60 joint transformer blocks (transformer_blocks.0..59)
  - No single_transformer_blocks
  - 9 safetensors shards (~40.9 GB total)

Strategy:
  - Resident on GPU: img_in, norm_out, proj_out, time_text_embed (small)
  - Streamed: transformer_blocks[i] one-by-one via LiveSeeker hooks
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
    return [k for k in seeker.weight_map.keys() if not k.startswith("transformer_blocks.")]


def _apply_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)
        else:
            set_module_tensor_to_device(model, name, device, value=tensor)


def _evict_params(model: nn.Module, param_names: List[str]):
    for name in param_names:
        set_module_tensor_to_device(model, name, "meta")


class QwenImageTransformer2DModelStreamer:
    """
    Wraps QwenImageTransformer2DModel for memory-efficient layer streaming.
    Streams 60 joint transformer blocks directly from the original HF shards.
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
            block._qwen_shard_name = shard_name
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = module._qwen_shard_name
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
        module._qwen_loaded_params = list(sd.keys())

        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name = self._shard_order[next_pos]
            next_keys = _get_layer_keys(self.seeker, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_qwen_loaded_params", []))
        module._qwen_loaded_params = []
        return output

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False
    ) -> "QwenImageTransformer2DModelStreamer":
        from diffusers import QwenImageTransformer2DModel

        transformer_dir = Path(transformer_dir)

        print("Step 1/3 -- Initializing LiveSeeker on Qwen-Image transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors across 9 shards.")

        print("\nStep 2/3 -- Instantiating QwenImageTransformer2DModel on meta device ...")
        with init_empty_weights():
            cfg = QwenImageTransformer2DModel.load_config(str(transformer_dir / "config.json"))
            model = QwenImageTransformer2DModel.from_config(cfg)
        model.eval()

        # Move any non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("\nStep 3/3 -- Loading resident Qwen transformer tensors to GPU ...")
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
        print("QwenImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards\n")
        return streamer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Forward cache_context if the model has it (needed by official pipeline)
    def cache_context(self, *args, **kwargs):
        return self.model.cache_context(*args, **kwargs)
