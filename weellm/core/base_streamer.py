"""
base_streamer.py -- Abstract base class for WeeLLM weight streamers.

A streamer wraps a PyTorch model and installs forward_pre_hook /
forward_hook on every "heavy" layer.  On each forward call:
  - _pre_hook  loads the layer's shard from disk (or cache) to GPU
  - _post_hook evicts the layer back to the meta device

This pattern lets the full model run with only ONE layer in VRAM at a time.

Implementing a new streamer
---------------------------
Subclass BaseStreamer and implement the three abstract methods.  See
weellm/models/flux2_klein/transformer_streamer.py for a reference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class BaseStreamer(ABC):
    """
    Abstract hook-based weight streamer.

    Concrete subclasses (e.g. FluxStreamer, UNetStreamer) handle the
    model-specific shard layout and loading logic, while sharing the
    same pre/post hook interface.
    """

    # ------------------------------------------------------------------
    # Factory (subclasses should implement this)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "BaseStreamer":
        """
        Build a streaming-enabled model from a directory.

        Implementations should:
        1. Split the monolithic checkpoint into per-layer shards (once).
        2. Instantiate the model on the ``meta`` device (zero VRAM).
        3. Load resident tensors (embeddings, norms) to GPU.
        4. Call _install_hooks().
        """
        ...

    # ------------------------------------------------------------------
    # Hook interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _install_hooks(self) -> None:
        """Register forward_pre_hook and forward_hook on each streamable layer."""
        ...

    @abstractmethod
    def _pre_hook(self, module: nn.Module, args) -> None:
        """
        Called immediately before a layer's forward pass.
        Load the layer's shard from disk/cache to GPU.
        """
        ...

    @abstractmethod
    def _post_hook(self, module: nn.Module, args, output):
        """
        Called immediately after a layer's forward pass.
        Evict the layer's parameters back to the meta device.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Delegate to the underlying model."""
        return self.model(*args, **kwargs)
