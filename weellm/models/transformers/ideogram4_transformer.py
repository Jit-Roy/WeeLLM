"""
ideogram4_transformer.py -- Hook-based layer-streaming for Ideogram 4 diffusion transformer.

Uses the Double-Stream buffer overlap to load the massive 34 FP8 blocks directly from the SSD
while maintaining under 4.0 GB VRAM footprint.

Note: Ideogram4 uses FP8 weight dequantization (weight * weight_scale). We override
`apply_state_dict` from BaseTransformerStreamer to perform the FP8 math on-the-fly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")


class Ideogram4Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based streaming wrapper for Ideogram4Transformer2DModel.
    Loads 34 FP8-quantized blocks one at a time from SSD, dequantizes them
    (weight * weight_scale → bfloat16) before placing on GPU, and evicts
    immediately after the forward pass.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        blocks = getattr(self.model, "layers", getattr(self.model, "transformer_blocks", []))
        prefix = "layers" if hasattr(self.model, "layers") else "transformer_blocks"
        for i, block in enumerate(blocks):
            order.append((f"{prefix}.{i}", block))
        return order

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not k.startswith("layers.") and not k.startswith("transformer_blocks.")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Override to apply FP8 dequantization (weight * scale) before placing on GPU."""
        processed_sd = {}
        for name, tensor in state_dict.items():
            if name.endswith(".weight_scale"):
                continue
            
            if name.endswith(".weight") and f"{name}_scale" in state_dict:
                scale = state_dict[f"{name}_scale"].to(device=tensor.device, dtype=torch.float32)
                
                if scale.dim() == 1:
                    if scale.numel() == tensor.shape[0]:
                        scale = scale.view(-1, 1)
                    elif len(tensor.shape) > 1 and scale.numel() == tensor.shape[1]:
                        scale = scale.view(1, -1)
                        
                tensor = (tensor.to(torch.float32) * scale).to(self.dtype)
                
            processed_sd[name] = tensor

        for name, tensor in processed_sd.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self.model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self.model, name, self.device, value=tensor)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "Ideogram4Transformer2DModelStreamer":
        import diffusers
        if not hasattr(diffusers, "Ideogram4Transformer2DModel"):
            raise ImportError(
                "Diffusers does not have Ideogram4Transformer2DModel. "
                "Please update diffusers: pip install -U diffusers"
            )

        model_dir = Path(model_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Ideogram4 Transformer weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("Instantiating Ideogram4Transformer2DModel on meta device ...")
        from accelerate import init_empty_weights
        from weellm.utils import default_dtype
        config = diffusers.Ideogram4Transformer2DModel.load_config(str(model_dir))
        with default_dtype(dtype), init_empty_weights():
            model = diffusers.Ideogram4Transformer2DModel.from_config(config)
        model.eval()

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        
        for name, buf in list(model.named_buffers()):
            if buf.device.type != "meta":
                set_module_tensor_to_device(
                    model, name, device,
                    value=buf.to(device, dtype=dtype if buf.is_floating_point() else None),
                )

        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d transformer blocks for streaming.",
            len(streamer._get_shard_order())
        )
        logger.info("Ideogram4Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
