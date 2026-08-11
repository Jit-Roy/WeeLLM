"""
hidream_transformer_2d_model.py -- Hook-based layer streaming for HiDreamImageTransformer2DModel.

Streams `double_stream_blocks` and `single_stream_blocks`.
Keeps embeddings and final layers resident on GPU.
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

_STREAMING_PREFIXES = ("double_stream_blocks.", "single_stream_blocks.")


class HiDreamImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps HiDreamImageTransformer2DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        for i, block in enumerate(self.model.double_stream_blocks):
            order.append((f"double_stream_blocks.{i}", block))
        for i, block in enumerate(self.model.single_stream_blocks):
            order.append((f"single_stream_blocks.{i}", block))
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
    ) -> "HiDreamImageTransformer2DModelStreamer":
        from diffusers import HiDreamImageTransformer2DModel
        from accelerate import init_empty_weights
        from accelerate.utils.modeling import set_module_tensor_to_device
        from weellm.utils import default_dtype

        model_dir = Path(model_dir)

        logger.info("Initializing SafetensorsLiveSeeker on HiDream Transformer weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("Instantiating HiDreamImageTransformer2DModel on meta device ...")
        config = HiDreamImageTransformer2DModel.load_config(model_dir)
        with default_dtype(dtype), init_empty_weights():
            model = HiDreamImageTransformer2DModel.from_config(config)
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
            "Installed %d double blocks + %d single blocks for streaming.",
            len(model.double_stream_blocks),
            len(model.single_stream_blocks),
        )
        logger.info("HiDreamImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
