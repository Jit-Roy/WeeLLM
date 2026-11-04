"""
wan_animate_2_transformer_3d_model.py -- Hook-based layer-streaming for Wan-Animate 3D Video Transformer.

Architecture:
  - WanAnimate2Transformer3DModel
  - 40 transformer blocks with in-context FlexAttention and dual-pass KV caching
  - Full safetensors shard streaming support

Strategy:
  - Resident on GPU: patch_embedding, text_embedding, time_embedding, time_projection, head, img_emb
  - Streamed: blocks.{0..N-1}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
from weellm.io.seeker import get_seeker
from weellm.io.utils import default_dtype, clean_memory, report_memory

logger = logging.getLogger("weellm")

# Safetensors diffusers naming (blocks.N.*)
_CKPT_STREAMING_PREFIX = "blocks."


class WanAnimate2Transformer3DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based layer streamer for WanAnimate2Transformer3DModel.
    Streams transformer blocks sequentially during the denoising forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Never pin 14B blocks in VRAM: prevents OOM across multi-segment video passes
        self._cache_budget_bytes = 0
        self._calibration_done = True
        self._pinned_blocks = set()

    def _post_hook(self, module: nn.Module, args, output):
        from weellm.models.transformers.base_transformer_streamer import _SHARD_NAME_ATTR
        shard_name = getattr(module, _SHARD_NAME_ATTR, "")
        pos = self._shard_name_to_pos.get(shard_name, -1)
        output = super()._post_hook(module, args, output)
        # NOTE: torch.cuda.empty_cache() intentionally removed — calling it after
        # every block (40× per step) forces the CUDA allocator to defragment its pool
        # on each call, causing per-step time to grow linearly with step count.
        # The base-class evict_module() already sets parameters to meta, so VRAM is
        # freed correctly. empty_cache() is only needed after the *full* forward pass.
        compute_t = getattr(module, "_weellm_last_compute_t", None)
        if compute_t is not None:
            logger.info(
                "    [Transformer] Block %s (%d/%d) complete (evicted) | compute=%.2fs",
                shard_name, pos + 1, len(self._shard_order), compute_t,
            )
        else:
            logger.info(
                "    [Transformer] Block %s (%d/%d) complete (evicted)",
                shard_name, pos + 1, len(self._shard_order),
            )
        return output

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        if hasattr(self.model, "blocks"):
            for i, block in enumerate(self.model.blocks):
                order.append((f"blocks.{i}", block))
        return order

    def _get_resident_ckpt_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not k.startswith(_CKPT_STREAMING_PREFIX)
        ]

    def _get_resident_keys(self) -> List[str]:
        return self._get_resident_ckpt_keys()

    def _ckpt_shard_name(self, diffusers_shard_name: str) -> str:
        return diffusers_shard_name

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        ckpt_name = self._ckpt_shard_name(shard_name)
        return [
            k for k in self.seeker.weight_map
            if k.startswith(ckpt_name + ".")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        from weellm.io.memory import place_tensors
        place_tensors(self.model, state_dict, self.device, self.dtype, skip_errors=skip_errors)


    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        prefetch_device: Optional[str] = None,
        cache_to_ram: bool = False,
    ) -> "WanAnimate2Transformer3DModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on WanAnimate transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        from accelerate import init_empty_weights
        from accelerate.utils.modeling import set_module_tensor_to_device

        from diffusers import WanAnimate2Transformer3DModel as TransformerModelCls

        logger.info("Step 2/3 -- Instantiating %s on meta device ...", TransformerModelCls.__name__)
        config = TransformerModelCls.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = TransformerModelCls.from_config(config)
        model.eval()

        # Move any non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf.to(device))

        logger.info("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
        streamer = cls(
            model=model,
            seeker=seeker,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
            prefetch_device=prefetch_device,
        )
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        if resident_ckpt_keys:
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device=device, dtype=dtype)
            streamer.apply_state_dict(raw_sd, skip_errors=True)
            del raw_sd

        clean_memory(device)
        report_memory("After resident load")

        block_count = len(streamer._get_shard_order())
        logger.info("Installed %d blocks for streaming.", block_count)
        logger.info("%sStreamer ready. Mode: Live Seek from original shards", TransformerModelCls.__name__)
        return streamer


