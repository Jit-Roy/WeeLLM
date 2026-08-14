"""
ernie_image_transformer_2d_model.py -- Hook-based layer-streaming for ErnieImageTransformer2DModel.

Architecture:
  - N shared AdaLN blocks (layers.0..N-1)

Strategy:
  - Resident on GPU: x_embedder, text_proj, time_proj, time_embedding, pos_embed, adaLN_modulation, final_norm, final_linear
  - Streamed: layers[i]
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

_STREAMING_PREFIXES = ("layers.",)


class ErnieImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps ErnieImageTransformer2DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
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
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "ErnieImageTransformer2DModelStreamer":
        try:
            from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel
        except ImportError:
            raise ImportError(
                "ErnieImageTransformer2DModel requires diffusers 0.40.0.dev0 or higher. "
                "Please run `pip install git+https://github.com/huggingface/diffusers`."
            )

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(ErnieImageTransformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d shared AdaLN blocks for streaming.",
            len(model.layers),
        )
        logger.info("ErnieImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
