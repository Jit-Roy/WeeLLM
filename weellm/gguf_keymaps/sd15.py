"""
sd15.py -- GGUF key map for Stable Diffusion 1.5 format.

Detected by: "model.diffusion_model." prefix WITHOUT SDXL markers.
"""
from typing import Any, Dict, List

from weellm.gguf_keymaps.sdxl import RESNET_TEMPLATE, ATTN_TEMPLATE, _expand_unet_blocks

# ── SD 1.5 block layout ───────────────────────────────────────────────────────
# (LDM prefix, Diffusers prefix, num_transformer_blocks, is_resnet)
_SD15_BLOCKS = [
    ("input_blocks.1.0",   "down_blocks.0.resnets.0",    0, True),
    ("input_blocks.1.1",   "down_blocks.0.attentions.0", 1, False),
    ("input_blocks.2.0",   "down_blocks.0.resnets.1",    0, True),
    ("input_blocks.2.1",   "down_blocks.0.attentions.1", 1, False),
    ("input_blocks.4.0",   "down_blocks.1.resnets.0",    0, True),
    ("input_blocks.4.1",   "down_blocks.1.attentions.0", 1, False),
    ("input_blocks.5.0",   "down_blocks.1.resnets.1",    0, True),
    ("input_blocks.5.1",   "down_blocks.1.attentions.1", 1, False),
    ("input_blocks.7.0",   "down_blocks.2.resnets.0",    0, True),
    ("input_blocks.7.1",   "down_blocks.2.attentions.0", 1, False),
    ("input_blocks.8.0",   "down_blocks.2.resnets.1",    0, True),
    ("input_blocks.8.1",   "down_blocks.2.attentions.1", 1, False),
    ("input_blocks.10.0",  "down_blocks.3.resnets.0",    0, True),
    ("input_blocks.11.0",  "down_blocks.3.resnets.1",    0, True),
    ("middle_block.0",     "mid_block.resnets.0",         0, True),
    ("middle_block.1",     "mid_block.attentions.0",      1, False),
    ("middle_block.2",     "mid_block.resnets.1",         0, True),
    ("output_blocks.0.0",  "up_blocks.0.resnets.0",       0, True),
    ("output_blocks.1.0",  "up_blocks.0.resnets.1",       0, True),
    ("output_blocks.2.0",  "up_blocks.0.resnets.2",       0, True),
    ("output_blocks.3.0",  "up_blocks.1.resnets.0",       0, True),
    ("output_blocks.3.1",  "up_blocks.1.attentions.0",    1, False),
    ("output_blocks.4.0",  "up_blocks.1.resnets.1",       0, True),
    ("output_blocks.4.1",  "up_blocks.1.attentions.1",    1, False),
    ("output_blocks.5.0",  "up_blocks.1.resnets.2",       0, True),
    ("output_blocks.5.1",  "up_blocks.1.attentions.2",    1, False),
    ("output_blocks.6.0",  "up_blocks.2.resnets.0",       0, True),
    ("output_blocks.6.1",  "up_blocks.2.attentions.0",    1, False),
    ("output_blocks.7.0",  "up_blocks.2.resnets.1",       0, True),
    ("output_blocks.7.1",  "up_blocks.2.attentions.1",    1, False),
    ("output_blocks.8.0",  "up_blocks.2.resnets.2",       0, True),
    ("output_blocks.8.1",  "up_blocks.2.attentions.2",    1, False),
    ("output_blocks.9.0",  "up_blocks.3.resnets.0",       0, True),
    ("output_blocks.9.1",  "up_blocks.3.attentions.0",    1, False),
    ("output_blocks.10.0", "up_blocks.3.resnets.1",       0, True),
    ("output_blocks.10.1", "up_blocks.3.attentions.1",    1, False),
    ("output_blocks.11.0", "up_blocks.3.resnets.2",       0, True),
    ("output_blocks.11.1", "up_blocks.3.attentions.2",    1, False),
]

_SD15_BASE_MAP = {
    "model.diffusion_model.time_embed.0.weight":           "time_embedding.linear_1.weight",
    "model.diffusion_model.time_embed.0.bias":             "time_embedding.linear_1.bias",
    "model.diffusion_model.time_embed.2.weight":           "time_embedding.linear_2.weight",
    "model.diffusion_model.time_embed.2.bias":             "time_embedding.linear_2.bias",
    "model.diffusion_model.input_blocks.0.0.weight":       "conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.bias":         "conv_in.bias",
    "model.diffusion_model.out.0.weight":                  "conv_norm_out.weight",
    "model.diffusion_model.out.0.bias":                    "conv_norm_out.bias",
    "model.diffusion_model.out.2.weight":                  "conv_out.weight",
    "model.diffusion_model.out.2.bias":                    "conv_out.bias",
    "model.diffusion_model.input_blocks.3.0.op.weight":    "down_blocks.0.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias":      "down_blocks.0.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight":    "down_blocks.1.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias":      "down_blocks.1.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.9.0.op.weight":    "down_blocks.2.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.9.0.op.bias":      "down_blocks.2.downsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.2.1.conv.weight": "up_blocks.0.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.2.1.conv.bias":   "up_blocks.0.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "up_blocks.1.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias":   "up_blocks.1.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "up_blocks.2.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias":   "up_blocks.2.upsamplers.0.conv.bias",
}


class SD15KeyMap:
    NAME = "sd15"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # Must have the diffusion_model prefix but NOT be SDXL
        return (
            any(k.startswith("model.diffusion_model.") for k in gguf_keys)
            and not any("label_emb." in k for k in gguf_keys)
            and not any("transformer_blocks.9." in k for k in gguf_keys)
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        flat = _expand_unet_blocks(_SD15_BASE_MAP, _SD15_BLOCKS)
        return {k: [(v, None)] for k, v in flat.items()}
