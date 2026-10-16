"""
safetensors_base.py -- Shared base class for WeeLLM safetensors seekers.

Provides the common _DTYPE_MAP and _parse_index logic used by both
SafetensorsDiskSeeker and SafetensorsRAMSeeker, eliminating duplication
and ensuring both implementations stay in sync.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np

# ---------------------------------------------------------------------------
# Unified dtype map — shared by both disk and RAM seekers
# ---------------------------------------------------------------------------

DTYPE_MAP: Dict[str, type] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,      # stored as uint16; reinterpreted as bfloat16 after load
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8":  np.int8,
    "U8":  np.uint8,
    "BOOL": np.bool_,
    "F8_E4M3": np.uint8,    # stored as uint8; reinterpreted as float8_e4m3fn after load
}


class SafetensorsBase:
    """
    Shared base for SafetensorsDiskSeeker and SafetensorsRAMSeeker.

    Handles index parsing (model.safetensors.index.json or single-shard fallback)
    and the dtype/shape metadata lookups. Subclasses implement ``get_tensors()``.
    """

    def __init__(self, model_dir: Union[str, Path]):
        path = Path(model_dir)
        if path.is_file() and path.suffix == ".safetensors":
            self.model_dir = path.parent
            self._explicit_file = path.name
        else:
            self.model_dir = path
            self._explicit_file = None

        # Public weight_map always contains *diffusers* keys after _parse_index.
        self.weight_map: Dict[str, str] = {}
        # Internal maps for raw-key lookups (header access, comfy expansion).
        # If no keymap applies, _d2s is identity and _keymap_cls is None.
        self._d2s: Dict[str, str] = {}          # diffusers_key -> src (raw checkpoint) key
        self._keymap_cls = None                  # detected keymap plugin, or None
        self._parsed_headers: Dict[str, dict] = {}
        self._data_bases: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Index parsing — shared logic
    # ------------------------------------------------------------------

    def _parse_index(self) -> None:
        """Populate ``self.weight_map`` from a sharded index or a single file."""
        if self._explicit_file:
            src_file = self._explicit_file
            header, _ = self._read_header(self.model_dir / src_file)
            raw_map = {
                k: src_file
                for k in header.keys()
                if k != "__metadata__"
            }
        else:
            index_path     = self.model_dir / "model.safetensors.index.json"
            alt_index_path = self.model_dir / "diffusion_pytorch_model.safetensors.index.json"

            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)["weight_map"]
            elif alt_index_path.exists():
                with open(alt_index_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)["weight_map"]
            else:
                src_file = self._find_single_shard()
                header, _ = self._read_header(self.model_dir / src_file)
                raw_map = {
                    k: src_file
                    for k in header.keys()
                    if k != "__metadata__"
                }

        self._apply_keymap(raw_map)

    def _apply_keymap(self, raw_map: Dict[str, str]) -> None:
        """Apply a safetensors keymap (if any) and populate self.weight_map / self._d2s."""
        from weellm.io.safetensors.keymaps import build_remap_fn
        remap_fn, keymap_cls = build_remap_fn(list(raw_map.keys()))
        self._keymap_cls = keymap_cls

        if remap_fn is None:
            # No remapping needed — diffusers keys == raw keys
            self.weight_map = raw_map
            self._d2s = {k: k for k in raw_map}   # identity
        else:
            # Build remapped weight_map (diffusers keys → shard filename)
            # and _d2s (diffusers key → (original raw key, slice_info))
            new_map: Dict[str, str] = {}
            d2s: Dict[str, Any] = {}
            for src_key, shard in raw_map.items():
                dst = remap_fn(src_key)
                if isinstance(dst, list):
                    for dst_key, slice_info in dst:
                        new_map[dst_key] = shard
                        d2s[dst_key] = (src_key, slice_info)
                else:
                    new_map[dst] = shard
                    d2s[dst] = (src_key, None)
            self.weight_map = new_map
            self._d2s = d2s

    def _find_single_shard(self) -> str:
        """Return the filename of the single safetensors shard in model_dir.

        Checks preferred canonical names first for determinism, then falls back
        to globbing for any ``.safetensors`` file in the directory.
        """
        preferred = [
            "model.safetensors",
            "model.fp16.safetensors",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ]
        for name in preferred:
            if (self.model_dir / name).exists():
                return name

        # Fall back: find any single .safetensors shard in the directory.
        found = sorted(self.model_dir.glob("*.safetensors"))
        if len(found) == 1:
            return found[0].name
        if len(found) > 1:
            raise FileNotFoundError(
                f"Multiple .safetensors files found in {self.model_dir} but no index file. "
                f"Cannot determine which shard to use: {[f.name for f in found]}"
            )
        raise FileNotFoundError(
            f"Could not find a safetensors index or any .safetensors shard in {self.model_dir}. "
            f"Preferred names checked: {preferred}"
        )

    # ------------------------------------------------------------------
    # Header parsing — shared logic
    # ------------------------------------------------------------------

    def _read_header(self, filepath: Path):
        """Parse and cache the safetensors header of *filepath*.

        Returns ``(header_dict, data_base_offset)`` where *data_base_offset*
        is the byte offset at which tensor data begins.
        """
        filename = filepath.name
        if filename in self._parsed_headers:
            return self._parsed_headers[filename], self._data_bases[filename]

        with open(filepath, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ValueError(f"File too small to be a safetensors file: {filepath}")
            header_size = struct.unpack("<Q", raw)[0]
            header      = json.loads(f.read(header_size).decode("utf-8"))
            data_base   = 8 + header_size

        self._parsed_headers[filename] = header
        self._data_bases[filename]     = data_base
        return header, data_base

    def get_block_bytes(self, keys: List[str]) -> int:
        """Return the total byte footprint of *keys* using cached header metadata.

        Reads shard headers (tiny JSON, cached after first access) but never
        loads actual tensor data. Safe to call from any seeker subclass.
        Used by streamers to compute adaptive prefetch depth without I/O.
        """
        total = 0
        for key in keys:
            if key not in self.weight_map:
                continue
            src     = self.weight_map[key]
            src_key = self._d2s.get(key, key)   # translate diffusers key → raw key
            header, _ = self._read_header(self.model_dir / src)
            if src_key not in header or src_key == "__metadata__":
                continue
            meta   = header[src_key]
            shape  = meta.get("shape", [])
            dstr   = meta.get("dtype", "F32")
            count  = int(np.prod(shape)) if shape else 1
            nbytes = count * np.dtype(DTYPE_MAP.get(dstr, np.uint8)).itemsize
            total += nbytes
        return total or 1   # never return 0 — prevents ZeroDivisionError in prefetch depth calc
