"""
joy_image_edit_transformer_3d_model.py -- Hook-based layer-streaming for JoyImageEditTransformer3DModel.

Architecture (JoyAI-Image-Edit):
  - 40 double transformer blocks (double_blocks.0..39)
  - 7 safetensors shards (~32.5 GB total)

Strategy:
  - Resident on GPU: img_in, condition_embedder, norm_out, proj_out (small)
  - Streamed: double_blocks[i] one-by-one via LiveSeeker hooks
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

_STREAMING_PREFIXES = ("double_blocks.",)


class JoyImageEditTransformer3DModelStreamer(BaseTransformerStreamer):
    """
    Wraps JoyImageEditTransformer3DModel for memory-efficient layer streaming.
    Streams 40 double transformer blocks directly from the original HF shards.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        return [
            (f"double_blocks.{i}", block)
            for i, block in enumerate(self.model.double_blocks)
        ]

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
    ) -> "JoyImageEditTransformer3DModelStreamer":
        from diffusers import JoyImageEditTransformer3DModel

        transformer_dir = Path(transformer_dir)

        logger.debug("Step 1/3 -- Initializing LiveSeeker on JoyImageEdit transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.debug("  Found %d tensors across shards.", len(seeker.weight_map))

        model = cls._load_model_on_meta(JoyImageEditTransformer3DModel, transformer_dir, device, dtype, seeker)

        logger.debug("Step 3/3 -- Loading resident JoyImageEdit transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.debug(
            "Installed %d double transformer blocks for streaming.", len(model.double_blocks)
        )

        def _cast_hidden_states(module, args, kwargs):
            if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                if kwargs["hidden_states"].is_floating_point() and kwargs["hidden_states"].dtype != streamer.dtype:
                    kwargs["hidden_states"] = kwargs["hidden_states"].to(streamer.dtype)
            elif len(args) > 0 and torch.is_tensor(args[0]):
                if args[0].is_floating_point() and args[0].dtype != streamer.dtype:
                    args = (args[0].to(streamer.dtype),) + args[1:]
            return args, kwargs

        model.register_forward_pre_hook(_cast_hidden_states, with_kwargs=True)

        logger.debug("JoyImageEditTransformer3DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
