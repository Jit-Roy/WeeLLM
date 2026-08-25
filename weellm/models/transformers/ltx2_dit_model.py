"""
ltx2_dit_model.py -- Hook-based layer-streaming for LTX-2.5 22B Video Transformer.

Architecture:
  - LTX2VideoTransformer3DModel
  - 39GB parameters
  - 48 transformer blocks

Strategy:
  - Resident on GPU: video_embeddings_connector, norm_out, scale_shift_tables
  - Streamed: transformer_blocks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import default_dtype, clean_memory, report_memory

logger = logging.getLogger("weellm")

# In the new diffusers pack, keys are directly named
_CKPT_STREAMING_PREFIX = "transformer_blocks."

def _remap_ckpt_key(ckpt_key: str) -> str:
    """Apply top-level prefix remapping from checkpoint → diffusers attribute names."""
    # Remove the comfy wrapper prefix if it exists (for backward compatibility)
    if ckpt_key.startswith("model.diffusion_model."):
        return ckpt_key[len("model.diffusion_model."):]
    return ckpt_key

class LTX2DiTModelStreamer(BaseTransformerStreamer):
    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        if hasattr(self.model, "transformer_blocks"):
            for i, block in enumerate(self.model.transformer_blocks):
                order.append((f"transformer_blocks.{i}", block))
        return order

    def _get_resident_ckpt_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not k.startswith(_CKPT_STREAMING_PREFIX)
        ]

    def _get_resident_keys(self) -> List[str]:
        return self._get_resident_ckpt_keys()

    def _ckpt_shard_name(self, diffusers_shard_name: str) -> str:
        # In the new diffusers format, the diffusers_shard_name IS the ckpt_shard_name
        return diffusers_shard_name

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        ckpt_name = self._ckpt_shard_name(shard_name)
        return [
            k for k in self.seeker.weight_map
            if k.startswith(ckpt_name + ".")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        remapped: Dict[str, torch.Tensor] = {}
        for ck, tensor in state_dict.items():
            dk = _remap_ckpt_key(ck)
            remapped[dk] = tensor

        from weellm.memory import place_tensors
        place_tensors(self.model, remapped, self.device, self.dtype, skip_errors=skip_errors)

    def _pre_hook(self, module: nn.Module, args):
        args = super()._pre_hook(module, args)
        
        from weellm.models.base_streamer import _SHARD_NAME_ATTR
        shard_name: str = getattr(module, _SHARD_NAME_ATTR)
        
        if hasattr(self, "lora_loader") and self.lora_loader is not None:
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
    ) -> "LTX2DiTModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on LTX-2.5 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        from diffusers import LTX2VideoTransformer3DModel
        from accelerate import init_empty_weights
        
        logger.info("  Instantiating LTX2VideoTransformer3DModel on meta device ...")
        
        # Load the full config from the local checkpoint directory.
        # Previously we used a hardcoded partial config which caused shape mismatches.
        config = LTX2VideoTransformer3DModel.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = LTX2VideoTransformer3DModel.from_config(config)
        model.eval()

        logger.info("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch, prefetch_device=prefetch_device)
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        if resident_ckpt_keys:
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device=device, dtype=dtype)
            remapped_sd = {_remap_ckpt_key(k): v for k, v in raw_sd.items()}
            streamer.apply_state_dict(remapped_sd, skip_errors=True)
            del remapped_sd

        clean_memory(device)
        report_memory("After resident load")

        block_count = len(streamer._get_shard_order())
        logger.info("Installed %d blocks for streaming.", block_count)
        logger.info("LTX2DiTModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
