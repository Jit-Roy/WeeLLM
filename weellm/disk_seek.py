"""
disk_seek.py -- Disk-based safetensors tensor streamer.

Reads specific tensors directly from disk using file seeks, completely
avoiding memory-mapping and duplicate shard copies. Optimal for NVMe SSDs
and local hardware.
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

    Completely avoids memory-mapping and loading duplicate shards. Uses a
    single shared byte buffer that grows on demand and is periodically
    released to avoid holding large amounts of RAM indefinitely.

    Best for: local machines with NVMe SSDs.
    """

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        # Per-instance buffer to avoid global state; not shared across threads.
        self._shared_buffer = bytearray()
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
        Dict mapping tensor name → ``torch.Tensor``.
        """
        # Group keys by source shard file to minimise file-open overhead.
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            src = self.weight_map[key]
            by_src.setdefault(src, []).append(key)

        result: Dict[str, torch.Tensor] = {}

        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)

            with open(filepath, "rb") as f:
                for key in src_keys:
                    meta      = header[key]
                    dtype_str = meta["dtype"]
                    shape     = meta["shape"]
                    start, end = meta["data_offsets"]

                    np_dtype = DTYPE_MAP[dtype_str]
                    count    = int(np.prod(shape)) if shape else 1
                    nbytes   = count * np.dtype(np_dtype).itemsize

                    # Grow the shared buffer on demand.
                    if len(self._shared_buffer) < nbytes:
                        self._shared_buffer = bytearray(
                            max(nbytes, len(self._shared_buffer) * 2)
                        )

                    view       = memoryview(self._shared_buffer)[:nbytes]
                    f.seek(data_base + start)
                    bytes_read = f.readinto(view)
                    if bytes_read != nbytes:
                        raise ValueError(
                            f"Short read for tensor '{key}' in {filepath}: "
                            f"expected {nbytes} bytes, got {bytes_read}"
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
                    # Cloning ensures the tensor owns its memory after the
                    # shared buffer is potentially reused on the next call.
                    if t.device.type == "cpu":
                        t = t.clone()

                    if dtype is not None and t.dtype != dtype and t.is_floating_point():
                        t = t.to(dtype=dtype)

                    result[key] = t

        # Release the shared buffer when it grows excessively large to avoid
        # holding hundreds of MB of RAM between small loads.
        if len(self._shared_buffer) > 128 * 1024 * 1024:
            self._shared_buffer = bytearray()

        return result
