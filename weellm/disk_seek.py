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
        self._parse_index()
        
        self._ram_cache: Dict[str, torch.Tensor] = {}
        self._ram_budget_bytes: int = 0
        self._ram_used_bytes: int = 0
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            # Leave a 4GB safety margin on System RAM
            self._ram_budget_bytes = mem.available - (4 * 1024 * 1024 * 1024)
            if self._ram_budget_bytes < 0:
                self._ram_budget_bytes = 0
            logger.info(f"DiskSeeker RAM Cache Budget: {self._ram_budget_bytes/1e9:.2f}GB")
        except ImportError:
            logger.warning("psutil not installed, disabling CPU RAM caching.")

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
        # Group missing keys by source shard file
        by_src: Dict[str, List[str]] = {}
        result: Dict[str, torch.Tensor] = {}
        missing_keys = []
        
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            
            # Fast path: Serve from RAM cache if available
            if key in self._ram_cache:
                t = self._ram_cache[key]
                t = t.to(device=device)
                if dtype is not None and t.dtype != dtype and t.is_floating_point():
                    t = t.to(dtype=dtype)
                result[key] = t
            else:
                missing_keys.append(key)
                src = self.weight_map[key]
                by_src.setdefault(src, []).append(key)

        if not missing_keys:
            return result

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

                    # Allocate a fresh bytearray for each tensor.
                    # This avoids the need for t.clone() later, cutting peak RAM in half!
                    buffer = bytearray(nbytes)
                    view = memoryview(buffer)
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
                        
                    # Attempt to cache the tensor in RAM
                    block_size = t.numel() * t.element_size()
                    if block_size <= self._ram_budget_bytes - self._ram_used_bytes:
                        try:
                            # Optional: Pin memory for faster H2D transfers if CUDA is available
                            if torch.cuda.is_available():
                                self._ram_cache[key] = t.pin_memory()
                            else:
                                self._ram_cache[key] = t
                        except Exception:
                            self._ram_cache[key] = t
                            
                        self._ram_used_bytes += block_size

                    t = t.to(device=device)

                    if dtype is not None and t.dtype != dtype and t.is_floating_point():
                        t = t.to(dtype=dtype)

                    result[key] = t

        return result
