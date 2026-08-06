import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

_DTYPE_MAP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16,
    "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
    "F8_E4M3": np.uint8,
}

class SafetensorsDiskSeeker:
    """
    Reads specific tensors from Hugging Face safetensors files directly from disk 
    using file seeks, completely avoiding memory-mapping and duplicate shards.
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.weight_map: Dict[str, str] = {}
        self._parsed_headers: Dict[str, dict] = {}
        self._data_bases: Dict[str, int] = {}
        
        self._parse_index()

    def _parse_index(self):
        index_path = self.model_dir / "model.safetensors.index.json"
        alt_index_path = self.model_dir / "diffusion_pytorch_model.safetensors.index.json"
        
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        elif alt_index_path.exists():
            with open(alt_index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single_st = self.model_dir / "model.safetensors"
            single_fp16 = self.model_dir / "model.fp16.safetensors"
            alt_single_st = self.model_dir / "diffusion_pytorch_model.safetensors"
            alt_fp16 = self.model_dir / "diffusion_pytorch_model.fp16.safetensors"
            
            if single_st.exists():
                src_file = single_st.name
            elif single_fp16.exists():
                src_file = single_fp16.name
            elif alt_single_st.exists():
                src_file = alt_single_st.name
            elif alt_fp16.exists():
                src_file = alt_fp16.name
            else:
                raise FileNotFoundError(f"Could not find safetensors index or single file in {self.model_dir}")
            
            header, _ = self._read_header(self.model_dir / src_file)
            for k in header.keys():
                if k != "__metadata__":
                    self.weight_map[k] = src_file

    def _read_header(self, filepath: Path) -> Tuple[dict, int]:
        filename = filepath.name
        if filename in self._parsed_headers:
            return self._parsed_headers[filename], self._data_bases[filename]

        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise ValueError(f"File {filepath} is too small to be a safetensors file.")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))
            data_base = 8 + header_size

        self._parsed_headers[filename] = header
        self._data_bases[filename] = data_base
        return header, data_base

    def get_tensors(self, keys: List[str], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            src = self.weight_map[key]
            if src not in by_src:
                by_src[src] = []
            by_src[src].append(key)

        result = {}

        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)

            with open(filepath, "rb") as f:
                for key in src_keys:
                    meta = header[key]
                    dtype_str = meta["dtype"]
                    shape = meta["shape"]
                    start, end = meta["data_offsets"]
                    
                    f.seek(data_base + start)
                    np_dtype = _DTYPE_MAP[dtype_str]
                    count = int(np.prod(shape)) if shape else 1
                    arr = np.empty(count, dtype=np_dtype)
                    bytes_read = f.readinto(arr.view(np.uint8))
                    if bytes_read != arr.nbytes:
                        raise ValueError(f"Failed to read tensor {key} from {filepath}")

                    if shape:
                        arr = arr.reshape(shape)

                    t = torch.from_numpy(arr)
                    if dtype_str == "BF16":
                        t = t.view(torch.bfloat16)
                    elif dtype_str == "F8_E4M3":
                        t = t.view(torch.float8_e4m3fn)

                    t = t.to(device=device)
                    if dtype is not None and t.dtype != dtype:
                        if t.is_floating_point():
                            t = t.to(dtype=dtype)

                    result[key] = t

        return result
