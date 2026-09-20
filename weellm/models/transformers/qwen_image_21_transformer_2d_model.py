"""
qwen_image_21_transformer_2d_model.py -- Hook-based layer-streaming for QwenImage21Transformer2DModel.

This is for Qwen-Image 2.1, which uses a single-stream block-causal transformer.

Architecture (Qwen-Image 2.1):
  - 32 single-stream transformer blocks (transformer_blocks.0..31)
  - Joint text/image sequence with block-causal attention

Strategy:
  - Resident on GPU: pos_embed, time_text_embed, txt_in, img_in, modulation,
                     norm_out, proj_out  (small, always needed)
  - Streamed: transformer_blocks[i] one-by-one via LiveSeeker hooks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from weellm.io.seeker import get_seeker
from weellm.io.utils import clean_memory, report_memory
from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer

logger = logging.getLogger("weellm")

_STREAMING_PREFIXES = ("transformer_blocks.",)


def _import_transformer_cls():
    """Import QwenImage21Transformer2DModel with an actionable error message.

    The class landed in diffusers via huggingface/diffusers#14804 and is only
    available from 0.41.0 onwards, while WeeLLM's requirements only pin
    ``diffusers>=0.40.0`` for every other architecture.
    """
    try:
        from diffusers import QwenImage21Transformer2DModel
    except ImportError as exc:
        import diffusers

        raise ImportError(
            "Qwen-Image 2.1 requires QwenImage21Transformer2DModel, which is only available "
            f"in diffusers >= 0.41.0 (installed: {getattr(diffusers, '__version__', 'unknown')}). "
            "Install a build that contains it:\n"
            "    pip install -U git+https://github.com/huggingface/diffusers"
        ) from exc
    return QwenImage21Transformer2DModel


class QwenImage21Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps QwenImage21Transformer2DModel for memory-efficient layer streaming.
    Streams the single-stream transformer blocks directly from the original HF shards.
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

    # Forward cache_context if the model has it — QwenImage21Pipeline wraps each
    # conditional/unconditional transformer call in `transformer.cache_context(...)`.
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
    ) -> "QwenImage21Transformer2DModelStreamer":
        transformer_cls = _import_transformer_cls()

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Qwen-Image 2.1 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(transformer_cls, transformer_dir, device, dtype, seeker)

        logger.info("Step 3/3 -- Loading resident Qwen-Image 2.1 transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d single-stream transformer blocks for streaming.",
            len(model.transformer_blocks),
        )
        logger.info("QwenImage21Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
