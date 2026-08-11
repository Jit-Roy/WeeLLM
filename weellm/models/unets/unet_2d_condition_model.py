"""
unet_2d_condition_model.py -- Hook-based block-streaming for UNet2DConditionModel (SDXL / SD1.5).

Strategy:
  - Resident on GPU: conv_in, time_embedding, add_embedding, conv_norm_out, conv_out
  - Streamed:        down_blocks, mid_block, up_blocks  (loaded just-in-time, evicted after)
"""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from weellm.utils import default_dtype
from accelerate.utils.modeling import set_module_tensor_to_device
from diffusers import UNet2DConditionModel

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

_STREAMING_PREFIXES = ("down_blocks.", "mid_block.", "up_blocks.")


def _patch_forward_input_device(model: nn.Module) -> None:
    """Move UNet forward inputs onto the model device before execution."""
    original_forward = model.forward

    def patched_forward(self_obj, *args, **kwargs):
        try:
            model_device = next(self_obj.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        args = list(args)
        for i, value in enumerate(args[:4]):
            if torch.is_tensor(value) and value.device != model_device:
                args[i] = value.to(model_device)
        for key, value in list(kwargs.items()):
            if torch.is_tensor(value) and value.device != model_device:
                kwargs[key] = value.to(model_device)
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)


class UNet2DConditionModelStreamer(BaseTransformerStreamer):
    """
    Wraps UNet2DConditionModel for memory-efficient block streaming.

    Streams directly from original safetensors shards via SafetensorsDiskSeeker.
    Resident: conv_in, time_embedding, add_embedding, conv_norm_out, conv_out.
    Streamed: down_blocks, mid_block, up_blocks.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order: List[Tuple[str, nn.Module]] = []
        for i, block in enumerate(self.model.down_blocks):
            order.append((f"down_blocks.{i}", block))
        if hasattr(self.model, "mid_block") and self.model.mid_block is not None:
            order.append(("mid_block", self.model.mid_block))
        for i, block in enumerate(self.model.up_blocks):
            order.append((f"up_blocks.{i}", block))
        return order

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _STREAMING_PREFIXES)
        ]

    # Override _get_layer_keys because UNet uses prefix+"."-style matching
    # but mid_block has no trailing dot index.
    def _get_layer_keys(self, shard_name: str) -> List[str]:
        prefix = shard_name + "."
        return [k for k in self.seeker.weight_map if k.startswith(prefix)]

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "UNet2DConditionModelStreamer":
        path = os.path.join(model_dir, "unet")

        logger.info("Step 1/3 -- Initializing LiveSeeker on UNet weights ...")
        seeker = get_seeker(path, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors.", len(seeker.weight_map))

        logger.info("Step 2/3 -- Instantiating UNet2DConditionModel on meta device ...")
        config = UNet2DConditionModel.load_config(os.path.join(path, "config.json"))
        with default_dtype(dtype), init_empty_weights():
            model = UNet2DConditionModel.from_config(config)
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        logger.info("Step 3/3 -- Loading resident UNet tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        _patch_forward_input_device(model)
        report_memory("After resident load")

        logger.info(
            "Installed %d down blocks, 1 mid block, %d up blocks for streaming.",
            len(model.down_blocks),
            len(model.up_blocks),
        )
        logger.info("UNetStreamer ready. Mode: Live Seek from original shards")
        return streamer
