"""
splitter.py -- One-time splitting of the ZImageTransformer2DModel's sharded
safetensors checkpoint into per-layer shards for streaming.

Source layout (3 files, ~24 GB total):
    transformer/diffusion_pytorch_model-00001-of-00003.safetensors
    transformer/diffusion_pytorch_model-00002-of-00003.safetensors
    transformer/diffusion_pytorch_model-00003-of-00003.safetensors
    transformer/diffusion_pytorch_model.safetensors.index.json

Output layout:
    transformer/splitted_model/
        layers.0.safetensors  ...  layers.29.safetensors
        context_refiner.0.safetensors  context_refiner.1.safetensors
        noise_refiner.0.safetensors    noise_refiner.1.safetensors
        resident.safetensors

IMPORTANT: The actual splitting is run in a SUBPROCESS so that the 3 x 10 GB
memory-mapped files are fully released from virtual address space before the
main inference process begins.  On Windows, keeping these mappings alive in the
same process causes the OS to kill the process when inference begins.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_SHARD_DIR   = "splitted_model"
_DONE_SUFFIX = ".done"


def _shard_path(shard_dir: Path, shard_name: str) -> Path:
    return shard_dir / f"{shard_name}.safetensors"


def _shard_done(shard_dir: Path, shard_name: str) -> bool:
    return (
        _shard_path(shard_dir, shard_name).exists()
        and (shard_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").exists()
    )


def _classify_key(key: str):
    """Return shard_name for any weight key."""
    for prefix in ("layers.", "context_refiner.", "noise_refiner."):
        if key.startswith(prefix):
            idx = int(key.split(".")[1])
            return f"{prefix.rstrip('.')}.{idx}"
    return "resident"


# ---------------------------------------------------------------------------
# Subprocess worker -- called via `python -m` or `python splitter.py`
# This runs in isolation and exits cleanly, releasing all memory.
# ---------------------------------------------------------------------------

def _do_split(transformer_dir: str, shard_dir: str, force: bool) -> None:
    """The actual split work. Runs inside the subprocess."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tqdm import tqdm

    t_dir     = Path(transformer_dir)
    s_dir     = Path(shard_dir)
    index_path = t_dir / "diffusion_pytorch_model.safetensors.index.json"

    print("Reading weight map from transformer index ...")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]

    # Classify each key
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key in weight_map:
        shard_to_keys[_classify_key(key)].append(key)
    all_shard_names = sorted(shard_to_keys.keys())
    print(f"Found {len(all_shard_names)} output shards.")

    # Validate source files exist
    for fname in sorted(set(weight_map.values())):
        src = t_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Source shard not found: {src}")

    print(f"\nSplitting transformer -> {s_dir}")
    print("(This runs once and is skipped on future invocations)\n")

    for shard_name in tqdm(all_shard_names, desc="Writing shards"):
        if not force and _shard_done(s_dir, shard_name):
            continue

        keys = shard_to_keys[shard_name]

        # Group keys by source file
        by_src: Dict[str, list] = defaultdict(list)
        for key in keys:
            by_src[weight_map[key]].append(key)

        # Open source files ONE AT A TIME, materializing tensors into RAM
        # via .clone() BEFORE closing the handle.
        # Without .clone(), tensors remain backed by the mmap'd file;
        # closing the handle then frees that memory, causing STATUS_ACCESS_VIOLATION
        # when save_file later tries to serialize them.
        state: Dict = {}
        for src_fname in sorted(by_src.keys()):
            src_keys = by_src[src_fname]
            src_path = t_dir / src_fname
            with safe_open(str(src_path), framework="pt") as fh:
                for key in src_keys:
                    # .clone() copies data into a plain CPU tensor — no mmap dependency
                    state[key] = fh.get_tensor(key).clone()
            # Explicit GC before opening next file to release mmap pages
            import gc; gc.collect()

        save_file(state, str(_shard_path(s_dir, shard_name)))
        (s_dir / f"{shard_name}.safetensors{_DONE_SUFFIX}").touch()
        del state
        import gc; gc.collect()

    print(f"\nSplit complete. Shards written to: {s_dir}")


# ---------------------------------------------------------------------------
# Public API -- called from transformer_streamer.py
# ---------------------------------------------------------------------------

def split_zimage_transformer(
    transformer_dir: str | Path,
    shard_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Split the ZImageTransformer2DModel checkpoint into per-layer shards.

    The actual work is run in a subprocess so that all 3 x ~10 GB
    memory-mapped source files are fully released before inference starts.

    Returns
    -------
    Path
        Absolute path to the shard directory.
    """
    transformer_dir = Path(transformer_dir).resolve()
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Shard index not found: {index_path}")

    if shard_dir is None:
        shard_dir = transformer_dir / _SHARD_DIR
    shard_dir = Path(shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Quick-check: are all shards already done?
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_names = sorted({_classify_key(k) for k in weight_map})

    if not force and all(_shard_done(shard_dir, n) for n in shard_names):
        print(f"Shards already exist at {shard_dir} -- skipping split.")
        return shard_dir

    # Run splitting in a subprocess.
    # IMPORTANT: The script must NOT import anything from weellm — doing so
    # triggers weellm/__init__.py which pulls in diffusers/accelerate/torch CUDA,
    # any of which can cause STATUS_ACCESS_VIOLATION in the subprocess environment.
    # We inline the entire logic as a self-contained script instead.
    print("Launching split subprocess (isolated memory, no weellm imports) ...")

    _inline_script = f"""
import json, gc, struct
from collections import defaultdict
from pathlib import Path
from safetensors.torch import save_file
from tqdm import tqdm
import numpy as np
import torch

DONE_SUFFIX = ".done"

# safetensors dtype string -> numpy dtype (BF16 handled separately)
_DTYPE_MAP = {{
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "BF16": np.uint16,   # read as uint16, re-view as bfloat16 via torch
    "I64": np.int64, "I32": np.int32, "I16": np.int16,
    "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}}

def read_tensors_seekable(filepath, keys_to_read):
    \"\"\"
    Read specific tensors from a safetensors file using plain seek/read.
    Does NOT use mmap, so does NOT commit Windows page file for the full file.
    \"\"\"
    keys_needed = set(keys_to_read)
    result = {{}}
    with open(str(filepath), "rb") as f:
        # Parse header
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_base = 8 + header_size   # byte offset where tensor data begins
        for key, meta in header.items():
            if key not in keys_needed:
                continue
            dtype_str = meta["dtype"]
            shape    = meta["shape"]
            start, end = meta["data_offsets"]
            f.seek(data_base + start)
            raw = f.read(end - start)
            np_dtype = _DTYPE_MAP[dtype_str]
            arr = np.frombuffer(raw, dtype=np_dtype).copy()   # full copy, no mmap
            if shape:
                arr = arr.reshape(shape)
            t = torch.from_numpy(arr)
            if dtype_str == "BF16":
                t = t.view(torch.bfloat16)
            result[key] = t
    return result

def shard_path(shard_dir, shard_name):
    return Path(shard_dir) / f"{{shard_name}}.safetensors"

def shard_done(shard_dir, shard_name):
    p = shard_path(shard_dir, shard_name)
    return p.exists() and (Path(shard_dir) / f"{{shard_name}}.safetensors{{DONE_SUFFIX}}").exists()

def classify_key(key):
    for prefix in ("layers.", "context_refiner.", "noise_refiner."):
        if key.startswith(prefix):
            idx = int(key.split(".")[1])
            return f"{{prefix.rstrip('.')}}.{{idx}}"
    return "resident"

t_dir = Path(r"{transformer_dir}")
s_dir = Path(r"{shard_dir}")
force = {force}

index_path = t_dir / "diffusion_pytorch_model.safetensors.index.json"
with open(index_path, "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]

shard_to_keys = defaultdict(list)
for key in weight_map:
    shard_to_keys[classify_key(key)].append(key)
all_shard_names = sorted(shard_to_keys.keys())
print(f"Found {{len(all_shard_names)}} output shards.")

for fname in sorted(set(weight_map.values())):
    if not (t_dir / fname).exists():
        raise FileNotFoundError(f"Source shard not found: {{t_dir / fname}}")

print(f"\\nSplitting transformer -> {{s_dir}}")
print("(This runs once and is skipped on future invocations)\\n")

for shard_name in tqdm(all_shard_names, desc="Writing shards"):
    if not force and shard_done(s_dir, shard_name):
        continue
    keys = shard_to_keys[shard_name]
    by_src = defaultdict(list)
    for key in keys:
        by_src[weight_map[key]].append(key)

    state = {{}}
    for src_fname in sorted(by_src.keys()):
        # seek-based read: opens file, seeks to each tensor, reads raw bytes, closes.
        # NO mmap, NO page file commitment for the full 10 GB file.
        partial = read_tensors_seekable(t_dir / src_fname, by_src[src_fname])
        state.update(partial)
        del partial
        gc.collect()

    save_file(state, str(shard_path(s_dir, shard_name)))
    (s_dir / f"{{shard_name}}.safetensors{{DONE_SUFFIX}}").touch()
    del state
    gc.collect()

print(f"\\nSplit complete. Shards written to: {{s_dir}}")
"""

    result = subprocess.run(
        [sys.executable, "-c", _inline_script],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Transformer splitting subprocess failed with exit code {result.returncode}.\n"
            "Run the test again — partially-written shards have .done guards and will be skipped."
        )

    return shard_dir


def get_shard_path(shard_dir: Path, shard_name: str) -> Path:
    return _shard_path(shard_dir, shard_name)


# Allow: python splitter.py <transformer_dir> [shard_dir] [--force]
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("transformer_dir")
    p.add_argument("shard_dir", nargs="?", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    _do_split(args.transformer_dir, args.shard_dir or str(Path(args.transformer_dir) / _SHARD_DIR), args.force)
