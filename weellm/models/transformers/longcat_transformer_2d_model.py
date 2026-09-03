"""
longcat_transformer_2d_model.py -- Hook-based layer-streaming for LongCatImageTransformer2DModel.

Architecture (LongCat-Image):
  - 10 joint transformer blocks (transformer_blocks.0..9)
  - 20 single_transformer_blocks (single_transformer_blocks.0..19)

Strategy:
  - Resident on GPU: context_embedder, pos_embed, time_embed, x_embedder,
                     norm_out, proj_out  (small, always needed)
  - Streamed: transformer_blocks[i] and single_transformer_blocks[i]
              (loaded just-in-time via hooks, evicted after forward pass)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

_STREAMING_PREFIXES = ("transformer_blocks.", "single_transformer_blocks.")


class LongCatImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps LongCatImageTransformer2DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        for i, block in enumerate(self.model.transformer_blocks):
            order.append((f"transformer_blocks.{i}", block))
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
    ) -> "LongCatImageTransformer2DModelStreamer":
        from diffusers import LongCatImageTransformer2DModel

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on LongCat-Image transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors.", len(seeker.weight_map))

        model = cls._load_model_on_meta(LongCatImageTransformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident LongCat transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d joint and %d single transformer blocks for streaming.",
            len(model.transformer_blocks),
            len(model.single_transformer_blocks),
        )
        logger.info("LongCatImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
