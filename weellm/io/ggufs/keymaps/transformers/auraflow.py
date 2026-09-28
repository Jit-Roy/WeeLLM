"""
auraflow.py -- GGUF key map for AuraFlow (ComfyUI format).

ComfyUI renames AuraFlow tensors significantly:
  - joint_transformer_blocks -> double_layers
  - single_transformer_blocks -> single_layers
  - attn.to_q -> attn.w2q (joint), attn.w1q (single)
  - ff.linear_1 -> mlpX.c_fc1, etc.

Source of truth: comfy/utils.py::auraflow_to_diffusers()
"""
from typing import Any, Dict, List

# Joint/double block map: diffusers_key_suffix -> comfy_key_suffix
# (reversed from auraflow_to_diffusers which maps diffusers->comfy)
_JOINT_BLOCK_MAP = {
    "attn.w2q.weight":          "attn.to_q.weight",
    "attn.w2k.weight":          "attn.to_k.weight",
    "attn.w2v.weight":          "attn.to_v.weight",
    "attn.w2o.weight":          "attn.to_out.0.weight",
    "attn.w1q.weight":          "attn.add_q_proj.weight",
    "attn.w1k.weight":          "attn.add_k_proj.weight",
    "attn.w1v.weight":          "attn.add_v_proj.weight",
    "attn.w1o.weight":          "attn.to_add_out.weight",
    "mlpX.c_fc1.weight":        "ff.linear_1.weight",
    "mlpX.c_fc2.weight":        "ff.linear_2.weight",
    "mlpX.c_proj.weight":       "ff.out_projection.weight",
    "mlpC.c_fc1.weight":        "ff_context.linear_1.weight",
    "mlpC.c_fc2.weight":        "ff_context.linear_2.weight",
    "mlpC.c_proj.weight":       "ff_context.out_projection.weight",
    "modX.1.weight":            "norm1.linear.weight",
    "modC.1.weight":            "norm1_context.linear.weight",
}

# Single block map: diffusers_key_suffix -> comfy_key_suffix
_SINGLE_BLOCK_MAP = {
    "attn.w1q.weight":   "attn.to_q.weight",
    "attn.w1k.weight":   "attn.to_k.weight",
    "attn.w1v.weight":   "attn.to_v.weight",
    "attn.w1o.weight":   "attn.to_out.0.weight",
    "modCX.1.weight":    "norm1.linear.weight",
    "mlp.c_fc1.weight":  "ff.linear_1.weight",
    "mlp.c_fc2.weight":  "ff.linear_2.weight",
    "mlp.c_proj.weight": "ff.out_projection.weight",
}

# Global/basic keys: comfy_key -> diffusers_key
_MAP_BASIC = {
    "positional_encoding":          "pos_embed.pos_embed",
    "register_tokens":              "register_tokens",
    "t_embedder.mlp.0.weight":      "time_step_proj.linear_1.weight",
    "t_embedder.mlp.0.bias":        "time_step_proj.linear_1.bias",
    "t_embedder.mlp.2.weight":      "time_step_proj.linear_2.weight",
    "t_embedder.mlp.2.bias":        "time_step_proj.linear_2.bias",
    "cond_seq_linear.weight":       "context_embedder.weight",
    "init_x_linear.weight":         "pos_embed.proj.weight",
    "init_x_linear.bias":           "pos_embed.proj.bias",
    "final_linear.weight":          "proj_out.weight",
    "modF.1.weight":                "norm_out.linear.weight",
}


def _build_full_map(n_double: int, n_single: int) -> dict:
    m = dict(_MAP_BASIC)
    for i in range(n_double):
        for comfy_suffix, diffusers_suffix in _JOINT_BLOCK_MAP.items():
            m[f"double_layers.{i}.{comfy_suffix}"] = f"joint_transformer_blocks.{i}.{diffusers_suffix}"
    for i in range(n_single):
        for comfy_suffix, diffusers_suffix in _SINGLE_BLOCK_MAP.items():
            m[f"single_layers.{i}.{comfy_suffix}"] = f"single_transformer_blocks.{i}.{diffusers_suffix}"
    return m


class AuraFlowKeyMap:
    """
    Translates ComfyUI AuraFlow GGUF tensor names back to
    Hugging Face Diffusers AuraFlowTransformer2DModel names.
    """
    NAME = "auraflow"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        stripped = [k.replace("model.diffusion_model.", "") for k in gguf_keys]
        # Signature: uses ComfyUI-specific "double_layers" and "w2q" naming
        has_double_layers = any(k.startswith("double_layers.") for k in stripped)
        has_w2q = any("attn.w2q.weight" in k for k in stripped)
        return has_double_layers and has_w2q

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        # Count blocks dynamically from the actual file
        stripped = [k.replace("model.diffusion_model.", "") for k in gguf_keys]
        n_double = sum(1 for k in stripped if k.startswith("double_layers.") and k.endswith("attn.w2q.weight"))
        n_single = sum(1 for k in stripped if k.startswith("single_layers.") and k.endswith("attn.w1q.weight"))
        full_map = _build_full_map(n_double, n_single)

        mapping = {}
        for gguf_key in gguf_keys:
            clean = gguf_key.replace("model.diffusion_model.", "")
            mapping[gguf_key] = full_map.get(clean, clean)
        return mapping
