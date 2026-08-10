"""
ram_seek.py -- RAM-cached safetensors tensor streamer.

Loads entire safetensors shard files into CPU RAM as raw bytes objects and
serves individual tensors from memory. Bypasses disk I/O bottlenecks on slow
NAS/cloud drives (Kaggle, Colab). Use with ``--cache_to_ram``.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from weellm.safetensors_base import DTYPE_MAP, SafetensorsBase

logger = logging.getLogger("weellm")


class SafetensorsRAMSeeker(SafetensorsBase):
    """
    Reads specific tensors from Hugging Face safetensors files by eagerly
    loading the full shard into CPU RAM as a ``bytes`` object.

    This avoids repeated disk I/O on slow drives by serving all tensor reads
    directly from memory. Best for: cloud environments with slow disks
    (Kaggle, Colab).

    Call ``clear_ram_cache()`` to release the in-memory copy once inference
    is complete.
    """

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        # Maps shard filename → raw data bytes (excludes header, starts at data_base)
        self._ram_cache: Dict[str, bytes] = {}
        self._parse_index()

    # ------------------------------------------------------------------
    # RAM cache management
    # ------------------------------------------------------------------

    def _ensure_shard_cached(self, filepath: Path) -> None:
        """Load *filepath*'s data section into ``self._ram_cache`` if not cached."""
        import struct

        filename = filepath.name
        if filename in self._ram_cache:
            return

        logger.info("  [RAM Cache] Loading %s into CPU RAM...", filename)
        with open(filepath, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ValueError(f"File too small to be a safetensors file: {filepath}")
            header_size = struct.unpack("<Q", raw)[0]
            data_base   = 8 + header_size
            f.seek(data_base)
            self._ram_cache[filename] = f.read()

        # Ensure header is also parsed (re-uses base class caching)
        self._read_header(filepath)

    def clear_ram_cache(self) -> None:
        """Free the in-memory shard cache. Shards will be lazily reloaded on next access."""
        self._ram_cache.clear()
        gc.collect()

    # ------------------------------------------------------------------
    # Tensor loading
    # ------------------------------------------------------------------

    def _bytes_to_tensor(
        self,
        raw_bytes: bytes,
        dtype_str: str,
        shape: list,
        device: str,
        dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        np_dtype = DTYPE_MAP[dtype_str]
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).copy()
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
        return t

    def get_tensors(
        self,
        keys: List[str],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load the tensors named by *keys* from the in-memory RAM cache.

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
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            src = self.weight_map[key]
            by_src.setdefault(src, []).append(key)

        result: Dict[str, torch.Tensor] = {}

        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            self._ensure_shard_cached(filepath)
            header = self._parsed_headers[src_file]
            cache  = self._ram_cache[src_file]

            for key in src_keys:
                meta   = header[key]
                start, end = meta["data_offsets"]
                result[key] = self._bytes_to_tensor(
                    cache[start:end],
                    meta["dtype"],
                    meta["shape"],
                    device,
                    dtype,
                )

        return result
