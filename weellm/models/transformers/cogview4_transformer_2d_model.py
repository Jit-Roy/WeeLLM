"""
cogview4_transformer_2d_model.py -- Hook-based layer-streaming for CogView4Transformer2DModel.

Uses the Live Seek Architecture to load weights directly from Hugging Face shards
without creating any duplicated files on disk.
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

_STREAMING_PREFIXES = ("transformer_blocks.",)


class CogView4Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps CogView4Transformer2DModel (CogView4) for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        return [
            (f"transformer_blocks.{i}", block)
            for i, block in enumerate(self.model.transformer_blocks)
        ]

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _STREAMING_PREFIXES)
        ]

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "CogView4Transformer2DModelStreamer":
        from diffusers import CogView4Transformer2DModel

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(CogView4Transformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d transformer blocks for streaming.", len(model.transformer_blocks)
        )
        logger.info("CogView4Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
