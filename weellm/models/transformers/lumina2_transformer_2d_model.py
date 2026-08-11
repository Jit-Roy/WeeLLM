"""
lumina2_transformer_2d_model.py -- Hook-based streaming for Lumina2Transformer2DModel.

Lumina2 has context_refiner, noise_refiner, and layers as three separately streamed block groups.
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


class Lumina2Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based streaming wrapper for Lumina2Transformer2DModel.
    Streams context_refiner, noise_refiner, and main layer blocks on-demand.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        for i, block in enumerate(self.model.context_refiner):
            order.append((f"context_refiner.{i}", block))
        for i, block in enumerate(self.model.noise_refiner):
            order.append((f"noise_refiner.{i}", block))
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
        **kwargs,
    ) -> "Lumina2Transformer2DModelStreamer":
        from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel
        from accelerate import init_empty_weights
        from accelerate.utils.modeling import set_module_tensor_to_device
        from weellm.utils import default_dtype

        model_dir = Path(model_dir)

        logger.info("Initializing SafetensorsLiveSeeker on Lumina2 weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("Instantiating Lumina2Transformer2DModel on meta device ...")
        with default_dtype(dtype), init_empty_weights():
            model = Lumina2Transformer2DModel.from_config(
                str(model_dir), trust_remote_code=True
            )
        model.eval()

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        
        # Load non-meta buffers
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

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
        logger.info("Lumina2Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
