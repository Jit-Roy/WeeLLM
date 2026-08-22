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


# ---------------------------------------------------------------------------
# Module-level shared RAM cache — ONE pool shared across ALL seekers (VAE,
# text encoder, transformer).
#
# BUG FIX: Previously, each SafetensorsDiskSeeker calculated its own
# independent RAM budget from available system RAM. With 3 seekers, each saw
# e.g. 26 GB "available" and independently tried to cache up to that amount,
# resulting in 3x oversubscription and an OS-level RAM kill (silent OOM) on
# Kaggle. The fix is to calculate the budget ONCE and share a single dict
# across every seeker created during this process lifetime.
# ---------------------------------------------------------------------------
_RAM_CACHE: Dict[str, torch.Tensor] = {}
_RAM_USED_BYTES: int = 0
_RAM_BUDGET_BYTES: int = -1   # -1 = not yet initialised


def _init_global_ram_budget() -> None:
    """Calculate the shared global RAM budget exactly once, across all seekers."""
    global _RAM_BUDGET_BYTES
    if _RAM_BUDGET_BYTES != -1:
        return   # already initialised
    try:
        import psutil
        mem = psutil.virtual_memory()
        # Leave a 6 GB safety margin for OS, PyTorch activations, and other
        # process overhead. This shared budget is used by VAE + TE + transformer
        # combined — not per-seeker.
        budget = mem.available - (6 * 1024 * 1024 * 1024)
        _RAM_BUDGET_BYTES = max(0, budget)
        logger.info(
            "DiskSeeker Global RAM Cache Budget: %.2f GB  (system available: %.2f GB)",
            _RAM_BUDGET_BYTES / 1e9,
            mem.available / 1e9,
        )
    except ImportError:
        _RAM_BUDGET_BYTES = 0
        logger.warning("psutil not installed — RAM caching disabled.")


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
        # Initialise the shared global RAM budget (no-op after the first call).
        _init_global_ram_budget()

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
        global _RAM_CACHE, _RAM_USED_BYTES, _RAM_BUDGET_BYTES

        # Group missing keys by source shard file
        by_src: Dict[str, List[str]] = {}
        result: Dict[str, torch.Tensor] = {}
        missing_keys = []

        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")

            # Fast path: serve from shared global RAM cache if available.
            if key in _RAM_CACHE:
                t = _RAM_CACHE[key].to(device=device)
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
                    meta       = header[key]
                    dtype_str  = meta["dtype"]
                    shape      = meta["shape"]
                    start, end = meta["data_offsets"]

                    np_dtype = DTYPE_MAP[dtype_str]
                    count    = int(np.prod(shape)) if shape else 1
                    nbytes   = count * np.dtype(np_dtype).itemsize

                    # Allocate a fresh bytearray for each tensor.
                    # This avoids the need for t.clone() later, cutting peak RAM in half!
                    buffer = bytearray(nbytes)
                    view   = memoryview(buffer)
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

                    # Attempt to cache the tensor in the shared global RAM pool.
                    # _RAM_USED_BYTES is checked against the single shared budget so
                    # all seekers combined never exceed the allowed RAM.
                    block_size = t.numel() * t.element_size()
                    if (
                        _RAM_BUDGET_BYTES > 0
                        and _RAM_USED_BYTES + block_size <= _RAM_BUDGET_BYTES
                        and key not in _RAM_CACHE
                    ):
                        try:
                            cached = t.pin_memory() if torch.cuda.is_available() else t
                            _RAM_CACHE[key] = cached
                            _RAM_USED_BYTES += block_size
                        except Exception:
                            pass  # pinning failed — skip caching this tensor

                    t = t.to(device=device)

                    if dtype is not None and t.dtype != dtype and t.is_floating_point():
                        t = t.to(dtype=dtype)

                    result[key] = t

        return result
