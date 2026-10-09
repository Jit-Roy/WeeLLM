"""
disk_seek.py -- Disk-based safetensors tensor streamer.

Reads specific tensors directly from disk using file seeks, completely
avoiding memory-mapping and duplicate shard copies. Optimal for NVMe SSDs
and local hardware.

Design principle: this is a pure disk reader. All pipeline scheduling
decisions (when to read, what to cache, how deep to prefetch) belong in the
streamer layer above. This module just reads efficiently.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

logger = logging.getLogger("weellm")


class SafetensorsDiskSeeker(SafetensorsBase):
    """
    Reads specific tensors from Hugging Face safetensors files directly from
    disk using file seeks.

    Completely avoids memory-mapping and loading entire duplicate shards.
    Uses file.readinto() into a fresh per-tensor bytearray to avoid any
    need for .clone() and to keep peak RAM minimal.

    Best for: local machines with NVMe SSDs.
    """

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self._parse_index()

    def get_tensors(
        self,
        keys: List[str],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        process_comfy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load the tensors named by *keys* (diffusers key names) from disk.
        Transparently handles key remapping (via SafetensorsBase._d2s) and
        dequantization (comfy_quant / SVDQ).
        """
        from weellm.io.safetensors.comfy_dequant import expand_comfy_keys, process_comfy_tensors

        # Translate diffusers keys -> raw checkpoint keys (_d2s is identity when no keymap).
        raw_keys = [self._d2s.get(k, k) for k in keys]

        # Build raw weight_map view for comfy expansion (needs original key names).
        raw_weight_map = {v: self.weight_map[k] for k, v in self._d2s.items() if k in self.weight_map}

        # Expand raw keys to include comfy_quant / SVDQ side-tensors.
        raw_keys_expanded = expand_comfy_keys(raw_keys, raw_weight_map)

        # Group by source shard to minimise file-open overhead.
        by_src: Dict[str, List[str]] = {}
        for raw_key in raw_keys_expanded:
            if raw_key not in raw_weight_map:
                raise KeyError(f"Tensor '{raw_key}' not found in model index.")
            by_src.setdefault(raw_weight_map[raw_key], []).append(raw_key)

        # Read raw tensors from disk.
        raw_result: Dict[str, torch.Tensor] = {}
        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)

            with open(filepath, "rb") as f:
                for raw_key in src_keys:
                    meta      = header[raw_key]
                    dtype_str = meta["dtype"]
                    shape     = meta["shape"]
                    start, _  = meta["data_offsets"]

                    np_dtype = DTYPE_MAP.get(dtype_str, np.uint8)
                    count    = int(np.prod(shape)) if shape else 1
                    nbytes   = count * np.dtype(np_dtype).itemsize

                    # Fresh bytearray per tensor - no clone() needed, halves peak RAM.
                    buf  = bytearray(nbytes)
                    view = memoryview(buf)
                    f.seek(data_base + start)
                    n = f.readinto(view)
                    if n != nbytes:
                        raise ValueError(
                            f"Short read for '{raw_key}' in {filepath}: "
                            f"expected {nbytes} B, got {n} B"
                        )

                    arr = np.frombuffer(view, dtype=np_dtype)
                    if shape:
                        arr = arr.reshape(shape)

                    t = torch.from_numpy(arr)
                    if dtype_str == "BF16":
                        t = t.view(torch.bfloat16)
                    elif dtype_str == "F8_E4M3":
                        t = t.view(torch.float8_e4m3fn)

                    t = t.to(device=device)
                    if dtype is not None and t.dtype != dtype and t.is_floating_point():
                        t = t.to(dtype=dtype)

                    raw_result[raw_key] = t

        # Dequantize comfy_quant / SVDQ groups (operates on raw keys).
        if process_comfy:
            raw_result = process_comfy_tensors(raw_result)

        # Remap raw keys back to diffusers keys and apply any reshape/postprocess.
        if self._keymap_cls is None:
            return raw_result

        s2d = {v: k for k, v in self._d2s.items()}
        result: Dict[str, torch.Tensor] = {}
        for raw_key, tensor in raw_result.items():
            dst_key = s2d.get(raw_key, raw_key)
            tensor  = self._keymap_cls.postprocess_tensor(dst_key, tensor, raw_key)
            result[dst_key] = tensor
        return result

