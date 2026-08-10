"""
utils.py -- Memory utilities shared across all WeeLLM model implementations.
"""

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger("weellm")


def clean_memory(device: str = "cuda") -> None:
    """Free CPU and GPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    Otherwise, assumes it's a Hugging Face repo ID and downloads it via
    ``snapshot_download``.
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
        "Resolving '%s' via Hugging Face Hub (this may take a while if downloading) ...",
        model_id_or_path,
    )
    allow_patterns  = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]
    ignore_patterns = ["*.bin", "*.pt", "*.ckpt", "*.h5", "*.msgpack"]

    cached_path = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return Path(cached_path)
