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

from weellm.safetensors_base import DTYPE_MAP, SafetensorsBase

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
    ) -> Dict[str, torch.Tensor]:
        """
        Load the tensors named by *keys* from disk and return them as a dict.

        Parameters
        ----------
        keys:
            Tensor names to load (must be present in ``self.weight_map``).
        device:
            PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        dtype:
            If given, floating-point tensors are cast to this dtype after load.

        Returns
        -------
        Dict mapping tensor name -> ``torch.Tensor``.
        """
        # Group keys by source shard file to minimise file-open overhead.
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            by_src.setdefault(self.weight_map[key], []).append(key)

        result: Dict[str, torch.Tensor] = {}
        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)

            with open(filepath, "rb") as f:
                for key in src_keys:
                    meta       = header[key]
                    dtype_str  = meta["dtype"]
                    shape      = meta["shape"]
                    start, _   = meta["data_offsets"]

                    np_dtype = DTYPE_MAP[dtype_str]
                    count    = int(np.prod(shape)) if shape else 1
                    nbytes   = count * np.dtype(np_dtype).itemsize

                    # Fresh bytearray per tensor — no clone() needed, halves peak RAM.
                    buf  = bytearray(nbytes)
                    view = memoryview(buf)
                    f.seek(data_base + start)
                    n = f.readinto(view)
                    if n != nbytes:
                        raise ValueError(
                            f"Short read for '{key}' in {filepath}: "
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

                    result[key] = t

        return result
