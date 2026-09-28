"""
longcat.py -- GGUF key map for LongCat-Image (ComfyUI format).

ComfyUI stores LongCat weights using FLUX naming conventions
(img_in, txt_in, double_blocks, single_blocks, time_in).
This keymap translates them back to the Diffusers naming
used by WeeLLM (x_embedder, context_embedder, transformer_blocks, etc.).

Source of truth: comfy/utils.py::flux_to_diffusers() and MAP_BASIC
LongCat has: 10 double_blocks, 20 single_blocks
"""
from typing import Any, Dict, List

# MAP_BASIC from ComfyUI's flux_to_diffusers (comfy/utils.py lines 760-783)
# Format: comfy_gguf_key -> diffusers_key
_MAP_BASIC = {
    "final_layer.linear.bias":               "proj_out.bias",
    "final_layer.linear.weight":             "proj_out.weight",
    "img_in.bias":                           "x_embedder.bias",
    "img_in.weight":                         "x_embedder.weight",
    # NOTE: LongCat has no vec_in/guidance_in (vec_in_dim=None, guidance_embed=False)
    "time_in.in_layer.bias":                 "time_text_embed.timestep_embedder.linear_1.bias",
    "time_in.in_layer.weight":               "time_text_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.bias":                "time_text_embed.timestep_embedder.linear_2.bias",
    "time_in.out_layer.weight":              "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.bias":                           "context_embedder.bias",
    "txt_in.weight":                         "context_embedder.weight",
    # norm_out: ComfyUI applies swap_scale_shift but for loading we just map the key.
    # The actual weight transformation is handled by WeeLLM's dequant pipeline.
    "final_layer.adaLN_modulation.1.bias":   "norm_out.linear.bias",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
}

# double_blocks -> transformer_blocks (joint attention blocks)
# block_map from comfy/utils.py lines 700-729
_DOUBLE_BLOCK_MAP = {
    "attn.to_out.0.weight":          "attn.to_out.0.weight",
    "attn.to_out.0.bias":            "attn.to_out.0.bias",
    "norm1.linear.weight":           "norm1.linear.weight",
    "norm1.linear.bias":             "norm1.linear.bias",
    "norm1_context.linear.weight":   "norm1_context.linear.weight",
    "norm1_context.linear.bias":     "norm1_context.linear.bias",
    "attn.to_add_out.weight":        "attn.to_add_out.weight",
    "attn.to_add_out.bias":          "attn.to_add_out.bias",
    "ff.net.0.proj.weight":          "ff.net.0.proj.weight",
    "ff.net.0.proj.bias":            "ff.net.0.proj.bias",
    "ff.net.2.weight":               "ff.net.2.weight",
    "ff.net.2.bias":                 "ff.net.2.bias",
    "ff_context.net.0.proj.weight":  "ff_context.net.0.proj.weight",
    "ff_context.net.0.proj.bias":    "ff_context.net.0.proj.bias",
    "ff_context.net.2.weight":       "ff_context.net.2.weight",
    "ff_context.net.2.bias":         "ff_context.net.2.bias",
    "attn.norm_q.weight":            "attn.norm_q.weight",
    "attn.norm_k.weight":            "attn.norm_k.weight",
    "attn.norm_added_q.weight":      "attn.norm_added_q.weight",
    "attn.norm_added_k.weight":      "attn.norm_added_k.weight",
    # fused QKV (img_attn.qkv) is split into to_q, to_k, to_v by the GGUF dequant layer
    "img_attn.proj.weight":          "attn.to_out.0.weight",
    "img_attn.proj.bias":            "attn.to_out.0.bias",
    "img_mod.lin.weight":            "norm1.linear.weight",
    "img_mod.lin.bias":              "norm1.linear.bias",
    "txt_mod.lin.weight":            "norm1_context.linear.weight",
    "txt_mod.lin.bias":              "norm1_context.linear.bias",
    "txt_attn.proj.weight":          "attn.to_add_out.weight",
    "txt_attn.proj.bias":            "attn.to_add_out.bias",
    "img_mlp.0.weight":              "ff.net.0.proj.weight",
    "img_mlp.0.bias":                "ff.net.0.proj.bias",
    "img_mlp.2.weight":              "ff.net.2.weight",
    "img_mlp.2.bias":                "ff.net.2.bias",
    "txt_mlp.0.weight":              "ff_context.net.0.proj.weight",
    "txt_mlp.0.bias":                "ff_context.net.0.proj.bias",
    "txt_mlp.2.weight":              "ff_context.net.2.weight",
    "txt_mlp.2.bias":                "ff_context.net.2.bias",
    "img_attn.norm.query_norm.weight": "attn.norm_q.weight",
    "img_attn.norm.key_norm.weight":   "attn.norm_k.weight",
    "txt_attn.norm.query_norm.weight": "attn.norm_added_q.weight",
    "txt_attn.norm.key_norm.weight":   "attn.norm_added_k.weight",
}

# single_blocks -> single_transformer_blocks
# block_map from comfy/utils.py lines 746-755
_SINGLE_BLOCK_MAP = {
    "modulation.lin.weight":  "norm.linear.weight",
    "modulation.lin.bias":    "norm.linear.bias",
    "linear2.weight":         "proj_out.weight",
    "linear2.bias":           "proj_out.bias",
    "norm.query_norm.weight": "attn.norm_q.weight",
    "norm.key_norm.weight":   "attn.norm_k.weight",
}

# Build full forward map (comfy_name -> diffusers_name)
def _build_full_map():
    m = dict(_MAP_BASIC)
    for i in range(10):  # LongCat: 10 joint transformer blocks
        for comfy_suffix, diffusers_suffix in _DOUBLE_BLOCK_MAP.items():
            m[f"double_blocks.{i}.{comfy_suffix}"] = f"transformer_blocks.{i}.{diffusers_suffix}"
    for i in range(20):  # LongCat: 20 single transformer blocks
        for comfy_suffix, diffusers_suffix in _SINGLE_BLOCK_MAP.items():
            m[f"single_blocks.{i}.{comfy_suffix}"] = f"single_transformer_blocks.{i}.{diffusers_suffix}"
    return m

_FULL_MAP = _build_full_map()


class LongCatKeyMap:
    """
    Translates ComfyUI LongCat GGUF tensor names (FLUX-style) back to
    Hugging Face Diffusers LongCatImageTransformer2DModel names.
    """
    NAME = "longcat"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # ComfyUI prefix is optional (present if exported from ComfyUI model files)
        stripped = [k.replace("model.diffusion_model.", "") for k in gguf_keys]
        # Signature: uses FLUX-style img_in/txt_in but has NO vec_in (vec_in_dim=None)
        # and has exactly 10 double blocks (not 19 like FLUX)
        has_img_in = any(k == "img_in.weight" for k in stripped)
        has_txt_in = any(k == "txt_in.weight" for k in stripped)
        no_vec_in = not any(k.startswith("vector_in.") for k in stripped)
        has_10_double = any(k.startswith("double_blocks.9.") for k in stripped)
        no_11th_double = not any(k.startswith("double_blocks.10.") for k in stripped)
        return has_img_in and has_txt_in and no_vec_in and has_10_double and no_11th_double

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        mapping = {}
        for gguf_key in gguf_keys:
            clean = gguf_key.replace("model.diffusion_model.", "")
            mapping[gguf_key] = _FULL_MAP.get(clean, clean)
        return mapping
