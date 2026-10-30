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

from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

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

        logger.debug("  [RAM Cache] Loading %s into CPU RAM...", filename)
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
        process_comfy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load the tensors named by *keys* (diffusers key names) from the in-memory RAM cache.
        Transparently handles key remapping (via SafetensorsBase._d2s) and
        dequantization (comfy_quant / SVDQ).
        """
        from weellm.io.safetensors.comfy_dequant import expand_comfy_keys, process_comfy_tensors

        # Translate diffusers keys -> raw checkpoint keys (_d2s is identity when no keymap).
        raw_keys = []
        for k in keys:
            entry = self._d2s.get(k, k)
            raw_keys.append(entry[0] if isinstance(entry, (tuple, list)) else entry)

        # Use the underlying raw weight map so dequantization side-tensors (.comfy_quant,
        # .weight_scale, etc.) are always discoverable, even if stripped from Diffusers weight_map.
        raw_weight_map = getattr(self, "_raw_weight_map", None)
        if raw_weight_map is None:
            raw_weight_map = {}
            for k, v in self._d2s.items():
                if k in self.weight_map:
                    rk = v[0] if isinstance(v, (tuple, list)) else v
                    raw_weight_map[rk] = self.weight_map[k]

        # Expand raw keys to include comfy_quant / SVDQ side-tensors.
        raw_keys_expanded = expand_comfy_keys(raw_keys, raw_weight_map)

        by_src: Dict[str, List[str]] = {}
        for raw_key in raw_keys_expanded:
            if raw_key not in raw_weight_map:
                raise KeyError(f"Tensor '{raw_key}' not found in model index.")
            src = raw_weight_map[raw_key]
            by_src.setdefault(src, []).append(raw_key)

        raw_result: Dict[str, torch.Tensor] = {}

        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            self._ensure_shard_cached(filepath)
            header = self._parsed_headers[src_file]
            cache  = self._ram_cache[src_file]

            for raw_key in src_keys:
                meta   = header[raw_key]
                start, end = meta["data_offsets"]
                raw_result[raw_key] = self._bytes_to_tensor(
                    cache[start:end],
                    meta["dtype"],
                    meta["shape"],
                    device,
                    dtype,
                )

        # When a keymap is active, dequantisation MUST happen inside the seeker
        # (before remapping), regardless of the process_comfy flag — see disk_seek.py.
        effective_process_comfy = process_comfy or (self._keymap_cls is not None)

        # Dequantize comfy_quant / SVDQ groups (operates on raw keys).
        raw_result = process_comfy_tensors(raw_result) if effective_process_comfy else raw_result

        # Remap raw keys back to diffusers keys and apply any reshape/postprocess.
        if self._keymap_cls is None:
            return raw_result

        result: Dict[str, torch.Tensor] = {}
        for dst_key in keys:
            if dst_key not in self._d2s:
                if dst_key in raw_result:
                    result[dst_key] = raw_result[dst_key]
                continue
            entry = self._d2s[dst_key]
            if isinstance(entry, (tuple, list)):
                raw_key, slice_info = entry
            else:
                raw_key, slice_info = entry, None

            if raw_key not in raw_result:
                continue

            tensor = raw_result[raw_key]
            if slice_info is not None:
                split_idx, total_splits = slice_info
                chunk_size = tensor.shape[0] // total_splits
                tensor = tensor[split_idx * chunk_size : (split_idx + 1) * chunk_size].clone()

            if hasattr(self._keymap_cls, "postprocess_tensor"):
                tensor = self._keymap_cls.postprocess_tensor(dst_key, tensor, raw_key)
            result[dst_key] = tensor
        return result

