"""
utils.py -- Memory utilities shared across all WeeLLM model implementations.
"""

import gc
from pathlib import Path
import torch


def clean_memory(device: str = "cuda") -> None:
    """Free CPU and GPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def report_memory(tag: str = "") -> None:
    """Print current VRAM and RAM usage to stdout."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved()  / 1e9
        print(f"[{tag}] VRAM alloc={alloc:.3f} GB | reserved={reserved:.3f} GB")
    try:
        import psutil, os
        ram = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"[{tag}] RAM (process RSS) = {ram:.3f} GB")
    except ImportError:
        pass


def resolve_model_path(model_id_or_path: str) -> Path:
    """
    Resolves a model string to a local Path.
    If it's an existing local directory, returns it directly.
    Otherwise, assumes it's a Hugging Face repo ID and downloads it via snapshot_download.
    """
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        return path.resolve()
    
    # Try downloading from Hugging Face Hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models automatically.\n"
            "Install it via: pip install huggingface_hub"
        )
        
    print(f"Resolving '{model_id_or_path}' via Hugging Face Hub (this may take a while if downloading) ...")
    # Only download essential file types (safetensors, configs, tokenizer files)
    allow_patterns = [
        "*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"
    ]
    # We ignore standard PyTorch/TF bins to save space, assuming safetensors exists
    ignore_patterns = ["*.bin", "*.pt", "*.ckpt", "*.h5", "*.msgpack"]
    
    cached_path = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return Path(cached_path)
