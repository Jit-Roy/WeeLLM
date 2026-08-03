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
}

class SafetensorsRAMSeeker:
    """
    Reads specific tensors from Hugging Face safetensors files by eagerly loading
    the entire file into CPU RAM as a bytes object. This avoids disk I/O bottlenecks 
    on slow NAS/Cloud drives (Kaggle/Colab) by serving tensor bytes directly from memory.
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.weight_map: Dict[str, str] = {}
        self._parsed_headers: Dict[str, dict] = {}
        self._ram_cache: Dict[str, bytes] = {}
        
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
            
            header = self._read_header(self.model_dir / src_file)
            for k in header.keys():
                if k != "__metadata__":
                    self.weight_map[k] = src_file

    def _read_header(self, filepath: Path) -> dict:
        filename = filepath.name
        
        # Ensure it is buffered in RAM cache
        if filename not in self._ram_cache:
            print(f"    [RAM Cache] Loading {filename} into CPU RAM...")
            with open(filepath, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    raise ValueError(f"File {filepath} is too small to be a safetensors file.")
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                data_base = 8 + header_size
                f.seek(data_base)
                self._ram_cache[filename] = f.read()

        if filename in self._parsed_headers:
            return self._parsed_headers[filename]

        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))
            
        self._parsed_headers[filename] = header
        return header

    def clear_ram_cache(self):
        """Frees the RAM cache buffer. Bytes will be lazy-reloaded on next access."""
        self._ram_cache.clear()
        import gc
        gc.collect()

    def _bytes_to_tensor(self, raw_bytes: bytes, dtype_str: str, shape: list, device: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
        np_dtype = _DTYPE_MAP[dtype_str]
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).copy()
        if shape:
            arr = arr.reshape(shape)
        
        t = torch.from_numpy(arr)
        if dtype_str == "BF16":
            t = t.view(torch.bfloat16)
        
        t = t.to(device=device)
        if dtype is not None and t.dtype != dtype:
            if t.is_floating_point():
                t = t.to(dtype=dtype)
        return t

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
            header = self._read_header(filepath)
            cache = self._ram_cache[src_file]

            for key in src_keys:
                meta = header[key]
                start, end = meta["data_offsets"]
                raw_bytes = cache[start:end]
                result[key] = self._bytes_to_tensor(raw_bytes, meta["dtype"], meta["shape"], device, dtype)

        return result
