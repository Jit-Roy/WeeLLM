"""
wan_transformer_3d_model.py -- Hook-based layer-streaming for Wan 3D Video Transformer.

Architecture:
  - WanTransformer3DModel
  - 30 transformer blocks (for Wan2.2-TI2V-5B)
  - Full safetensors shard streaming support

Strategy:
  - Resident on GPU: patch_embedding, condition_embedder, rope, norm_out, proj_out, scale_shift_table
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

# Safetensors diffusers naming (also applies to GGUF after WanGGUFKeyMap remaps transformer_blocks.N.* -> blocks.N.*)
_CKPT_STREAMING_PREFIX = "blocks."


class WanTransformer3DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based layer streamer for WanTransformer3DModel.
    Streams transformer blocks sequentially during the denoising forward pass.
    Supports both safetensors (blocks.N.*) and GGUF (transformer_blocks.N.*) weight naming.
    """

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

    def _pre_hook(self, module: nn.Module, args):
        args = super()._pre_hook(module, args)
        
        from weellm.models.transformers.base_transformer_streamer import _SHARD_NAME_ATTR
        shard_name: str = getattr(module, _SHARD_NAME_ATTR, "")
        
        if hasattr(self, "lora_loader") and self.lora_loader is not None and shard_name:
            self.lora_loader.apply_to_module(module, shard_name)
            
        return args

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        prefetch_device: Optional[str] = None,
        cache_to_ram: bool = False,
    ) -> "WanTransformer3DModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.debug("Step 1/3 -- Initializing LiveSeeker on Wan transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.debug("  Found %d tensors across HF shards.", len(seeker.weight_map))

        from diffusers import WanTransformer3DModel
        from accelerate import init_empty_weights
        from accelerate.utils.modeling import set_module_tensor_to_device

        logger.debug("Step 2/3 -- Instantiating WanTransformer3DModel on meta device ...")
        config = WanTransformer3DModel.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = WanTransformer3DModel.from_config(config)
        model.eval()

        # Move any non-meta buffers (e.g., rope freqs_cos, freqs_sin) to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf.to(device))

        logger.debug("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
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
        logger.debug("Installed %d blocks for streaming.", block_count)
        logger.debug("WanTransformer3DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
