"""
safetensors_base.py -- Shared base class for WeeLLM safetensors seekers.

Provides the common _DTYPE_MAP and _parse_index logic used by both
SafetensorsDiskSeeker and SafetensorsRAMSeeker, eliminating duplication
and ensuring both implementations stay in sync.
"""

import json
import struct
from pathlib import Path
from typing import Dict, Union

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
        self.model_dir = Path(model_dir)
        self.weight_map: Dict[str, str] = {}
        self._parsed_headers: Dict[str, dict] = {}
        self._data_bases: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Index parsing — shared logic
    # ------------------------------------------------------------------

    def _parse_index(self) -> None:
        """Populate ``self.weight_map`` from a sharded index or a single file."""
        index_path     = self.model_dir / "model.safetensors.index.json"
        alt_index_path = self.model_dir / "diffusion_pytorch_model.safetensors.index.json"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        elif alt_index_path.exists():
            with open(alt_index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            src_file = self._find_single_shard()
            header, _ = self._read_header(self.model_dir / src_file)
            self.weight_map = {
                k: src_file
                for k in header.keys()
                if k != "__metadata__"
            }

    def _find_single_shard(self) -> str:
        """Return the filename of the single safetensors shard in model_dir."""
        candidates = [
            "model.safetensors",
            "model.fp16.safetensors",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ]
        for name in candidates:
            if (self.model_dir / name).exists():
                return name
        raise FileNotFoundError(
            f"Could not find a safetensors index or single shard in {self.model_dir}. "
            f"Checked: {candidates}"
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
