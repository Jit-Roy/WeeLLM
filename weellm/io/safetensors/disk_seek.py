"""
disk_seek.py -- Disk-based safetensors tensor streamer.

Reads specific tensors directly from disk using persistent mmap-backed shard
views. Avoids duplicate shard copies in Python heap and avoids re-opening
large mmap mappings repeatedly on Windows. Optimal for NVMe SSDs.

Design principle: this is a pure disk reader. All pipeline scheduling
decisions (when to read, what to cache, how deep to prefetch) belong in the
streamer layer above. This module just reads efficiently.
"""

import logging
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

# Mapping from safetensors dtype string to torch dtype for torch.frombuffer.
# torch.frombuffer reads raw bytes directly, so we use the "storage" dtype
# (the dtype the bytes are physically stored as, same as DTYPE_MAP entries).
_TORCH_RAW_DTYPE_MAP: Dict[str, "torch.dtype"] = {
    "F64":     torch.float64,
    "F32":     torch.float32,
    "F16":     torch.float16,
    "BF16":    torch.int16,    # stored as uint16 ≡ int16 bytes; view as bfloat16 after load
    "I64":     torch.int64,
    "I32":     torch.int32,
    "I16":     torch.int16,
    "I8":      torch.int8,
    "U8":      torch.uint8,
    "BOOL":    torch.uint8,
    "F8_E4M3": torch.uint8,   # stored as uint8; view as float8_e4m3fn after load
}

from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

logger = logging.getLogger("weellm")


class SafetensorsDiskSeeker(SafetensorsBase):
    """
    Reads specific tensors from Hugging Face safetensors files directly from
    disk using file seeks.

    Uses persistent mmap-backed shard views and zero-copy slices to minimize
    RAM overhead and avoid Windows kernel PTE exhaustion (WinError 1450).
    Provides explicit lifecycle cleanup via close() and __enter__/__exit__.

    Best for: local machines with NVMe SSDs.
    """

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self._parse_index()
        # Cache of open mmap objects keyed by str(filepath).
        # Avoids re-opening mmap on every get_tensors() call — repeated open/close
        # of large shard files (5 GB each) exhausts Windows kernel PTE resources
        # (WinError 1450) after ~200 calls across a multi-pass pipeline run.
        # Values: (file_handle, mmap_obj, memoryview) — all kept open for process lifetime.
        self._mmap_cache: Dict[str, tuple] = {}

    def close(self) -> None:
        """Close all cached memory mappings and file handles."""
        for f_handle, mm, mv in list(self._mmap_cache.values()):
            try:
                mv.release()
            except Exception:
                pass
            try:
                mm.close()
            except Exception:
                pass
            try:
                f_handle.close()
            except Exception:
                pass
        self._mmap_cache.clear()

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

            # Reuse a cached mmap for this shard. Opening a new mmap.mmap() for
            # every get_tensors() call (up to ~200 times per pipeline run on a 5 GB
            # shard file) exhausts Windows kernel page-table entries (WinError 1450).
            # Model weights are read-only and never modified, so it is safe to keep
            # the mmap open for the process lifetime.
            cache_key = str(filepath)
            if cache_key not in self._mmap_cache:
                f_handle = open(filepath, "rb")
                mm = mmap.mmap(f_handle.fileno(), 0, access=mmap.ACCESS_READ)
                mv = memoryview(mm)
                self._mmap_cache[cache_key] = (f_handle, mm, mv)
            _, mm, mm_view = self._mmap_cache[cache_key]

            for raw_key in src_keys:
                meta      = header[raw_key]
                dtype_str = meta["dtype"]
                shape     = meta["shape"]
                start, _  = meta["data_offsets"]

                count    = int(np.prod(shape)) if shape else 1
                raw_td   = _TORCH_RAW_DTYPE_MAP.get(dtype_str, torch.uint8)
                # BF16 and F8 are physically stored as uint16/uint8 with
                # element counts matching the original shape.
                if dtype_str == "BF16":
                    nbytes = count * 2   # uint16 = 2 bytes per element
                elif dtype_str in ("U8", "BOOL", "F8_E4M3", "I8"):
                    nbytes = count * 1
                else:
                    np_dtype = DTYPE_MAP.get(dtype_str, np.uint8)
                    nbytes = count * np.dtype(np_dtype).itemsize

                offset = data_base + start
                # Zero-copy view into mmap (OS page cache, not heap).
                # torch.frombuffer accepts read-only buffers (unlike torch.from_numpy).
                # del slice_mv and t_raw immediately after use to release the
                # exported pointer before any potential mmap lifecycle operations.
                slice_mv = mm_view[offset : offset + nbytes]
                t_raw = torch.frombuffer(slice_mv, dtype=raw_td)
                del slice_mv  # release mmap reference from this memoryview slice

                if device != "cpu":
                    # DMA from OS page cache → VRAM, no Python heap copy needed.
                    t = t_raw.to(device=device, non_blocking=False)
                else:
                    # Clone to heap so tensor can be used independently of mmap.
                    t = t_raw.clone()
                del t_raw  # release mmap-backed tensor

                # Reinterpret storage dtype as the logical tensor dtype.
                if dtype_str == "BF16":
                    t = t.view(torch.bfloat16)
                elif dtype_str == "F8_E4M3":
                    t = t.view(torch.float8_e4m3fn)
                elif dtype_str == "BOOL":
                    t = t.to(torch.bool)

                if shape:
                    t = t.reshape(shape)

                if dtype is not None and t.dtype != dtype and t.is_floating_point():
                    t = t.to(dtype=dtype)

                raw_result[raw_key] = t

        # When a keymap is active, dequantisation MUST happen inside the seeker
        # (before remapping), regardless of the process_comfy flag.
        # Reason: remapping renames .weight to its diffusers equivalent, but
        # side-tensors (.comfy_quant, .svdq_l1, etc.) have no diffusers mapping
        # and stay as raw keys.  If we skip dequantisation here, _do_h2d's
        # subsequent process_comfy_tensors() call sees mismatched key namespaces
        # and raises KeyError on the missing raw .weight key.
        effective_process_comfy = process_comfy or (self._keymap_cls is not None)

        # Dequantize comfy_quant / SVDQ groups (operates on raw keys).
        if effective_process_comfy:
            raw_result = process_comfy_tensors(raw_result)

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

