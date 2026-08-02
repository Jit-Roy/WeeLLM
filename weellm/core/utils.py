"""
utils.py -- Memory utilities shared across all WeeLLM model implementations.
"""

import gc
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
