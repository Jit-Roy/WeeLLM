"""
z_image_transformer_2d_model.py -- Hook-based layer-streaming for ZImageTransformer2DModel.

ZImageTransformer has an unusual architecture: noise_refiner, context_refiner,
and main layers.
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

_STREAMING_PREFIXES = ("context_refiner.", "noise_refiner.", "layers.")


class ZImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based weight streamer for ZImageTransformer2DModel.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        # Match the upstream diffusers forward pass order exactly:
        # 1) noise_refiner
        # 2) context_refiner
        # 3) main layers
        # Loading in the wrong sequence causes the streamed blocks to be evicted
        # and reloaded out of phase, which produces the repeated tile/block artifacts.
        order = []
        for i, block in enumerate(self.model.noise_refiner):
            order.append((f"noise_refiner.{i}", block))
        for i, block in enumerate(self.model.context_refiner):
            order.append((f"context_refiner.{i}", block))
        for i, block in enumerate(self.model.layers):
            order.append((f"layers.{i}", block))
        return order

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _STREAMING_PREFIXES)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "ZImageTransformer2DModelStreamer":
        from diffusers import ZImageTransformer2DModel

        model_dir = Path(model_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on ZImageTransformer weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(ZImageTransformer2DModel, model_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d transformer blocks for streaming.",
            len(model.context_refiner) + len(model.noise_refiner) + len(model.layers)
        )
        logger.info("ZImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
