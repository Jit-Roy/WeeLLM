"""
hidream_transformer_2d_model.py -- Hook-based layer streaming for HiDreamImageTransformer2DModel.

Streams `double_stream_blocks` and `single_stream_blocks`.
Keeps embeddings and final layers resident on GPU.

float32 memory optimisation
---------------------------
When dtype=float32 is requested (everything stays float32 — no dtype mixing):

  1. Prefetch is disabled automatically.  Normally the streamer pre-loads the
     next block's weights into VRAM in a background thread while the current
     block is computing, saving latency.  In float32 each block is ~0.9 GB vs
     ~0.45 GB in bfloat16, so the prefetch buffer alone adds ~0.9 GB of live
     VRAM.  Disabling it removes that pressure and keeps peak under 4 GB.

  2. Nothing else changes — weights, activations, and outputs are all float32.
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

_STREAMING_PREFIXES = ("double_stream_blocks.", "single_stream_blocks.", "caption_projection.")


class HiDreamImageTransformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps HiDreamImageTransformer2DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        if hasattr(self.model, "caption_projection") and self.model.caption_projection is not None:
            for i, proj in enumerate(self.model.caption_projection):
                order.append((f"caption_projection.{i}", proj))
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

        # ── float32 memory optimisation ────────────────────────────────────
        # In float32 mode each streaming block is ~0.9 GB vs ~0.45 GB in bf16.
        # Keeping the *prefetched* next block in VRAM simultaneously adds an
        # extra ~0.9 GB of live allocations and pushes the peak past 4 GB.
        # Disabling prefetch lets only one block sit in VRAM at a time.
        # Everything stays true float32 — no dtype mixing.
        if dtype == torch.float32 and prefetch:
            logger.info(
                "  [fp32] Disabling prefetch to save ~0.9 GB peak VRAM "
                "(one block at a time instead of two). Everything stays float32."
            )
            prefetch = False

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
            "Installed %d double blocks, %d single blocks, and %d caption projections for streaming.",
            len(model.double_stream_blocks),
            len(model.single_stream_blocks),
            len(getattr(model, "caption_projection", [])) if getattr(model, "caption_projection", None) is not None else 0
        )
        logger.info("HiDreamImageTransformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
