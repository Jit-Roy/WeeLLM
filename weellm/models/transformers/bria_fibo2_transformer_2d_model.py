"""
bria_fibo2_transformer_2d_model.py -- Hook-based layer-streaming for BriaFibo2Transformer2DModel.

BriaFibo2Transformer2DModel architecture:
- noise_refiner: ModuleList of BriaFibo2TransformerBlock (n_refiner_layers)
- perceiver: BriaFibo2PerceiverResampler with .layers (perceiver_num_layers)
- layers: ModuleList of BriaFibo2TransformerBlock (n_layers)
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

_STREAMING_PREFIXES = ("noise_refiner.", "perceiver.layers.", "layers.")


class BriaFibo2Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based weight streamer for BriaFibo2Transformer2DModel.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        for i, block in enumerate(self.model.noise_refiner):
            order.append((f"noise_refiner.{i}", block))
        for i, block in enumerate(self.model.perceiver.layers):
            order.append((f"perceiver.layers.{i}", block))
        for i, block in enumerate(self.model.layers):
            order.append((f"layers.{i}", block))
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
    ) -> "BriaFibo2Transformer2DModelStreamer":
        from diffusers import BriaFibo2Transformer2DModel

        transformer_dir = Path(transformer_dir)
        if (transformer_dir / "transformer").is_dir() and (transformer_dir / "transformer" / "config.json").exists():
            transformer_dir = transformer_dir / "transformer"

        logger.info("Step 1/3 -- Initializing LiveSeeker on BriaFibo2Transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(BriaFibo2Transformer2DModel, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)

        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d transformer blocks for streaming (%d noise refiner, %d perceiver, %d main layers).",
            len(model.noise_refiner) + len(model.perceiver.layers) + len(model.layers),
            len(model.noise_refiner),
            len(model.perceiver.layers),
            len(model.layers),
        )
        logger.info("BriaFibo2Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
