"""
qwen_image_2_1_transformer_2d_model.py -- Hook-based layer-streaming for QwenImage21Transformer2DModel.

Architecture (Qwen-Image-2.1):
  - 32 joint transformer blocks (transformer_blocks.0..31)
  - No single_transformer_blocks

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

from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
from weellm.io.seeker import get_seeker
from weellm.io.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

_STREAMING_PREFIXES = ("transformer_blocks.",)


class QwenImage21Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps QwenImage21Transformer2DModel for memory-efficient layer streaming.
    Streams 32 joint transformer blocks directly from the original HF shards.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        return [
            (f"transformer_blocks.{i}", block)
            for i, block in enumerate(self.model.transformer_blocks)
        ]

    def _get_resident_keys(self) -> List[str]:
        expected_keys = set(self.model.state_dict().keys())
        return [
            k for k in self.seeker.weight_map
            if k in expected_keys and not any(k.startswith(p) for p in _STREAMING_PREFIXES)
        ]

    # Forward cache_context if the model has it (needed by the official pipeline)
    def cache_context(self, *args, **kwargs):
        return self.model.cache_context(*args, **kwargs)

    def apply_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Override to unmerge gate_up before placing on GPU."""
        processed_sd = {}
        for name, tensor in state_dict.items():
            if "img_mlp.gate_up.weight" in name:
                base = name.replace("img_mlp.gate_up.weight", "img_mlp.")
                gate, proj = tensor.chunk(2, dim=0)
                processed_sd[base + "gate_layer.weight"] = gate
                processed_sd[base + "proj.weight"] = proj
            else:
                processed_sd[name] = tensor
        super().apply_state_dict(processed_sd)

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "QwenImage21Transformer2DModelStreamer":
        try:
            from diffusers import QwenImage21Transformer2DModel as QwenTransformer
        except ImportError:
            from diffusers import QwenImageTransformer2DModel as QwenTransformer

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Qwen-Image-2.1 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors.", len(seeker.weight_map))

        model = cls._load_model_on_meta(QwenTransformer, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident Qwen 2.1 transformer tensors to GPU ...")
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
        logger.info("QwenImage21Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
