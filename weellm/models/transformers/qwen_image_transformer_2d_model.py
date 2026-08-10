"""
qwen_image_transformer_2d_model.py -- Hook-based layer-streaming for QwenImageTransformer2DModel.

Architecture (Qwen-Image):
  - 60 joint transformer blocks (transformer_blocks.0..59)
  - No single_transformer_blocks
  - 9 safetensors shards (~40.9 GB total)

Strategy:
  - Resident on GPU: img_in, norm_out, proj_out, time_text_embed (small)
  - Streamed: transformer_blocks[i] one-by-one via LiveSeeker hooks
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


class QwenImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps QwenImageTransformer2DModel for memory-efficient layer streaming.
    Streams 60 joint transformer blocks directly from the original HF shards.
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

    # Forward cache_context if the model has it (needed by the official pipeline)
    def cache_context(self, *args, **kwargs):
        return self.model.cache_context(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "QwenImageTransformer2DModelStreamer":
        from diffusers import QwenImageTransformer2DModel

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Qwen-Image transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across 9 shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(QwenImageTransformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident Qwen transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d joint transformer blocks for streaming.", len(model.transformer_blocks)
        )
        logger.info("QwenImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
