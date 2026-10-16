"""
flux3_transformer_2d_model.py -- Hook-based layer-streaming for Flux3Transformer2DModel.

Architecture (FLUX 3 Image Klein):
  - 5 per-mode transformer blocks for text (txt_mode_blocks.0..4)
  - 5 per-mode transformer blocks for image (mode_blocks.image.0..4)
  - 5 per-mode transformer blocks for reference images (mode_blocks.image_cond.0..4)
  - 28 joint single transformer blocks (single_transformer_blocks.0..27)

Strategy:
  - Resident on GPU:
      x_embedders, context_embedder, time_proj, timestep_embedder, target_timestep_embedder,
      pos_embed, early_stream_modulations, single_stream_modulations, norm_out, proj_out
  - Streamed:
      txt_mode_blocks[i], mode_blocks.image[i], mode_blocks.image_cond[i], single_transformer_blocks[i]
      (loaded just-in-time via hooks, evicted after forward pass)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
from weellm.io.seeker import get_seeker
from weellm.io.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

_STREAMING_PREFIXES = ("txt_mode_blocks.", "mode_blocks.", "single_transformer_blocks.")


class Flux3Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps Flux3Transformer2DModel (FLUX.3 Image Klein) for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        # 1. Text mode blocks (executed first in forward pass)
        if hasattr(self.model, "txt_mode_blocks"):
            for i, block in enumerate(self.model.txt_mode_blocks):
                order.append((f"txt_mode_blocks.{i}", block))

        # 2. Image mode blocks
        if hasattr(self.model, "mode_blocks"):
            if "image" in self.model.mode_blocks:
                for i, block in enumerate(self.model.mode_blocks["image"]):
                    order.append((f"mode_blocks.image.{i}", block))
            # 3. Image condition mode blocks (for multi-reference / image editing)
            if "image_cond" in self.model.mode_blocks:
                for i, block in enumerate(self.model.mode_blocks["image_cond"]):
                    order.append((f"mode_blocks.image_cond.{i}", block))

        # 4. Joint single transformer blocks
        if hasattr(self.model, "single_transformer_blocks"):
            for i, block in enumerate(self.model.single_transformer_blocks):
                order.append((f"single_transformer_blocks.{i}", block))

        return order

    def _get_resident_keys(self) -> List[str]:
        expected_keys = set(self.model.state_dict().keys())
        return [
            k for k in self.seeker.weight_map
            if k in expected_keys and not any(k.startswith(p) for p in _STREAMING_PREFIXES)
        ]

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        **kwargs,
    ) -> "Flux3Transformer2DModelStreamer":
        try:
            from diffusers import Flux3Transformer2DModel
        except ImportError:
            try:
                from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
            except ImportError:
                raise ImportError(
                    "Flux3Transformer2DModel is required. Please install diffusers with FLUX 3 support."
                )

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Flux3 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(Flux3Transformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        num_txt_blocks = len(model.txt_mode_blocks) if hasattr(model, "txt_mode_blocks") else 0
        num_mode_blocks = sum(len(v) for v in model.mode_blocks.values()) if hasattr(model, "mode_blocks") else 0
        num_single_blocks = len(model.single_transformer_blocks) if hasattr(model, "single_transformer_blocks") else 0
        logger.info(
            "Installed %d txt blocks + %d mode blocks + %d single blocks for streaming.",
            num_txt_blocks,
            num_mode_blocks,
            num_single_blocks,
        )
        logger.info("Flux3Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
