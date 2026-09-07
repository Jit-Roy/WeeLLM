"""
sdxl.py -- GGUF key map for SDXL format.

Detected by: "model.diffusion_model." prefix AND ("label_emb." OR "transformer_blocks.9.").
Also exports shared UNet block templates used by sd15.py.
"""
from typing import Any, Dict, List

# ── Shared UNet block templates (used by both SDXL and SD 1.5) ────────────────

RESNET_TEMPLATE = {
    "in_layers.0.weight":   "norm1.weight",
    "in_layers.0.bias":     "norm1.bias",
    "in_layers.2.weight":   "conv1.weight",
    "in_layers.2.bias":     "conv1.bias",
    "out_layers.0.weight":  "norm2.weight",
    "out_layers.0.bias":    "norm2.bias",
    "out_layers.3.weight":  "conv2.weight",
    "out_layers.3.bias":    "conv2.bias",
    "emb_layers.1.weight":  "time_emb_proj.weight",
    "emb_layers.1.bias":    "time_emb_proj.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias":   "conv_shortcut.bias",
}

ATTN_TEMPLATE = {
    "norm.weight":      "norm.weight",
    "norm.bias":        "norm.bias",
    "proj_in.weight":   "proj_in.weight",
    "proj_in.bias":     "proj_in.bias",
    "proj_out.weight":  "proj_out.weight",
    "proj_out.bias":    "proj_out.bias",
    "transformer_blocks.{j}.norm1.weight":              "transformer_blocks.{j}.norm1.weight",
    "transformer_blocks.{j}.norm1.bias":                "transformer_blocks.{j}.norm1.bias",
    "transformer_blocks.{j}.attn1.to_q.weight":         "transformer_blocks.{j}.attn1.to_q.weight",
    "transformer_blocks.{j}.attn1.to_k.weight":         "transformer_blocks.{j}.attn1.to_k.weight",
    "transformer_blocks.{j}.attn1.to_v.weight":         "transformer_blocks.{j}.attn1.to_v.weight",
    "transformer_blocks.{j}.attn1.to_out.0.weight":     "transformer_blocks.{j}.attn1.to_out.0.weight",
    "transformer_blocks.{j}.attn1.to_out.0.bias":       "transformer_blocks.{j}.attn1.to_out.0.bias",
    "transformer_blocks.{j}.norm2.weight":              "transformer_blocks.{j}.norm2.weight",
    "transformer_blocks.{j}.norm2.bias":                "transformer_blocks.{j}.norm2.bias",
    "transformer_blocks.{j}.attn2.to_q.weight":         "transformer_blocks.{j}.attn2.to_q.weight",
    "transformer_blocks.{j}.attn2.to_k.weight":         "transformer_blocks.{j}.attn2.to_k.weight",
    "transformer_blocks.{j}.attn2.to_v.weight":         "transformer_blocks.{j}.attn2.to_v.weight",
    "transformer_blocks.{j}.attn2.to_out.0.weight":     "transformer_blocks.{j}.attn2.to_out.0.weight",
    "transformer_blocks.{j}.attn2.to_out.0.bias":       "transformer_blocks.{j}.attn2.to_out.0.bias",
    "transformer_blocks.{j}.norm3.weight":              "transformer_blocks.{j}.norm3.weight",
    "transformer_blocks.{j}.norm3.bias":                "transformer_blocks.{j}.norm3.bias",
    "transformer_blocks.{j}.ff.net.0.proj.weight":      "transformer_blocks.{j}.ff.net.0.proj.weight",
    "transformer_blocks.{j}.ff.net.0.proj.bias":        "transformer_blocks.{j}.ff.net.0.proj.bias",
    "transformer_blocks.{j}.ff.net.2.weight":           "transformer_blocks.{j}.ff.net.2.weight",
    "transformer_blocks.{j}.ff.net.2.bias":             "transformer_blocks.{j}.ff.net.2.bias",
}

# ── SDXL block layout ─────────────────────────────────────────────────────────
# (LDM prefix, Diffusers prefix, num_transformer_blocks, is_resnet)
_SDXL_BLOCKS = [
    ("input_blocks.1.0",  "down_blocks.0.resnets.0",     0,  True),
    ("input_blocks.2.0",  "down_blocks.0.resnets.1",     0,  True),
    ("input_blocks.4.0",  "down_blocks.1.resnets.0",     0,  True),
    ("input_blocks.4.1",  "down_blocks.1.attentions.0",  2,  False),
    ("input_blocks.5.0",  "down_blocks.1.resnets.1",     0,  True),
    ("input_blocks.5.1",  "down_blocks.1.attentions.1",  2,  False),
    ("input_blocks.7.0",  "down_blocks.2.resnets.0",     0,  True),
    ("input_blocks.7.1",  "down_blocks.2.attentions.0",  10, False),
    ("input_blocks.8.0",  "down_blocks.2.resnets.1",     0,  True),
    ("input_blocks.8.1",  "down_blocks.2.attentions.1",  10, False),
    ("middle_block.0",    "mid_block.resnets.0",          0,  True),
    ("middle_block.1",    "mid_block.attentions.0",       10, False),
    ("middle_block.2",    "mid_block.resnets.1",          0,  True),
    ("output_blocks.0.0", "up_blocks.0.resnets.0",        0,  True),
    ("output_blocks.0.1", "up_blocks.0.attentions.0",     10, False),
    ("output_blocks.1.0", "up_blocks.0.resnets.1",        0,  True),
    ("output_blocks.1.1", "up_blocks.0.attentions.1",     10, False),
    ("output_blocks.2.0", "up_blocks.0.resnets.2",        0,  True),
    ("output_blocks.2.1", "up_blocks.0.attentions.2",     10, False),
    ("output_blocks.3.0", "up_blocks.1.resnets.0",        0,  True),
    ("output_blocks.3.1", "up_blocks.1.attentions.0",     2,  False),
    ("output_blocks.4.0", "up_blocks.1.resnets.1",        0,  True),
    ("output_blocks.4.1", "up_blocks.1.attentions.1",     2,  False),
    ("output_blocks.5.0", "up_blocks.1.resnets.2",        0,  True),
    ("output_blocks.5.1", "up_blocks.1.attentions.2",     2,  False),
    ("output_blocks.6.0", "up_blocks.2.resnets.0",        0,  True),
    ("output_blocks.7.0", "up_blocks.2.resnets.1",        0,  True),
    ("output_blocks.8.0", "up_blocks.2.resnets.2",        0,  True),
]

_SDXL_BASE_MAP = {
    "model.diffusion_model.time_embed.0.weight":        "time_embedding.linear_1.weight",
    "model.diffusion_model.time_embed.0.bias":          "time_embedding.linear_1.bias",
    "model.diffusion_model.time_embed.2.weight":        "time_embedding.linear_2.weight",
    "model.diffusion_model.time_embed.2.bias":          "time_embedding.linear_2.bias",
    "model.diffusion_model.label_emb.0.0.weight":       "add_embedding.linear_1.weight",
    "model.diffusion_model.label_emb.0.0.bias":         "add_embedding.linear_1.bias",
    "model.diffusion_model.label_emb.0.2.weight":       "add_embedding.linear_2.weight",
    "model.diffusion_model.label_emb.0.2.bias":         "add_embedding.linear_2.bias",
    "model.diffusion_model.input_blocks.0.0.weight":    "conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.bias":      "conv_in.bias",
    "model.diffusion_model.out.0.weight":               "conv_norm_out.weight",
    "model.diffusion_model.out.0.bias":                 "conv_norm_out.bias",
    "model.diffusion_model.out.2.weight":               "conv_out.weight",
    "model.diffusion_model.out.2.bias":                 "conv_out.bias",
    # Downsamplers
    "model.diffusion_model.input_blocks.3.0.op.weight": "down_blocks.0.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias":   "down_blocks.0.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight": "down_blocks.1.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias":   "down_blocks.1.downsamplers.0.conv.bias",
    # Upsamplers
    "model.diffusion_model.output_blocks.2.2.conv.weight": "up_blocks.0.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.2.2.conv.bias":   "up_blocks.0.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "up_blocks.1.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias":   "up_blocks.1.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "up_blocks.2.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias":   "up_blocks.2.upsamplers.0.conv.bias",
}


def _expand_unet_blocks(base_map: Dict, blocks: list) -> Dict[str, str]:
    """Expand resnet/attention block templates into a flat key map."""
    remap = base_map.copy()
    for ldm_pref, diff_pref, num_t, is_resnet in blocks:
        template = RESNET_TEMPLATE if is_resnet else ATTN_TEMPLATE
        for k_src, k_dst in template.items():
            if "{j}" in k_src:
                for j in range(num_t):
                    remap[f"model.diffusion_model.{ldm_pref}.{k_src.replace('{j}', str(j))}"] = \
                        f"{diff_pref}.{k_dst.replace('{j}', str(j))}"
            else:
                remap[f"model.diffusion_model.{ldm_pref}.{k_src}"] = f"{diff_pref}.{k_dst}"
    return remap


class SDXLKeyMap:
    NAME = "sdxl"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        if not any(k.startswith("model.diffusion_model.") for k in gguf_keys):
            return False
        return (
            any("label_emb." in k for k in gguf_keys)
            or any("transformer_blocks.9." in k for k in gguf_keys)
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        flat = _expand_unet_blocks(_SDXL_BASE_MAP, _SDXL_BLOCKS)
        return {k: [(v, None)] for k, v in flat.items()}
