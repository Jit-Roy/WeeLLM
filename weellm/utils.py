"""
utils.py -- Memory utilities and model-path resolution shared across all WeeLLM components.
"""

import contextlib
import gc
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("weellm")

@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    """Temporarily set PyTorch's default dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def clean_memory(device: str = "cuda") -> None:
    """Free CPU and GPU memory aggressively.

    Uses a double-GC pattern:
    1. First gc.collect() drops Python references so tensors become candidates
       for CUDA freeing.
    2. cuda.empty_cache() + synchronize() returns those pages to the CUDA
       memory pool and flushes pending ops.
    3. Second gc.collect() picks up any __del__ finalizers triggered by CUDA
       cleanup itself (e.g. caching allocator callbacks).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def report_memory(tag: str = "") -> None:
    """Log current VRAM and RAM usage at DEBUG level."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved()  / 1e9
        logger.debug("[%s] VRAM alloc=%.3f GB | reserved=%.3f GB", tag, alloc, reserved)
    try:
        import psutil, os
        ram = psutil.Process(os.getpid()).memory_info().rss / 1e9
        logger.debug("[%s] RAM (process RSS) = %.3f GB", tag, ram)
    except ImportError:
        pass


def resolve_model_path(model_id_or_path: str) -> Path:
    """
    Resolve a model string to a local Path.

    If it's an existing local directory, returns it directly.

    Otherwise, assumes it's a Hugging Face repo ID and performs a smart
    component-only download: first fetches ``model_index.json`` to discover
    which subfolders are needed, then downloads only those subfolders rather
    than the entire repository.
    """
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models automatically.\n"
            "Install it via: pip install huggingface_hub"
        )

    logger.info(
        "Path '%s' not found locally. Attempting to download from Hugging Face Hub ...",
        model_id_or_path,
    )

    # Phase 1: fetch model_index.json to know which components we need.
    logger.info("  Fetching model_index.json from '%s' ...", model_id_or_path)
    index_dir  = snapshot_download(model_id_or_path, allow_patterns=["model_index.json"])
    index_path = Path(index_dir) / "model_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # Phase 2: download only the required component subfolders.
    allow_patterns = ["model_index.json"]
    for key, value in index_data.items():
        if isinstance(value, list) and len(value) == 2:
            allow_patterns.append(f"{key}/*")

    logger.info("  Downloading only required components: %s", allow_patterns)
    cached_path = snapshot_download(model_id_or_path, allow_patterns=allow_patterns)
    return Path(cached_path)
