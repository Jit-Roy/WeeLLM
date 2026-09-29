"""
longcat.py -- GGUF key map for LongCat-Image (ComfyUI format).

ComfyUI stores LongCat weights using FLUX naming conventions
(img_in, txt_in, double_blocks, single_blocks, time_in).
This keymap translates them back to the Diffusers naming
used by WeeLLM (x_embedder, context_embedder, transformer_blocks, etc.).

Source of truth:
  - ComfyUI naming: comfy/utils.py::flux_to_diffusers() MAP_BASIC
  - Diffusers naming: verified from actual safetensors state_dict keys
    (LongCat-Image-Edit/transformer/diffusion_pytorch_model.safetensors)
  IMPORTANT: LongCat uses 'time_embed' NOT 'time_text_embed' (which is FLUX).
  LongCat has: 10 double_blocks, 20 single_blocks
"""
from typing import Any, Dict, List

# Tensors stored as [shift | scale] in the ComfyUI/BFL checkpoint but Diffusers'
# AdaLayerNormContinuous.forward() reads them as [scale | shift].
# We must swap the two halves, matching the same logic used by FluxKeyMap.
_SWAP_SCALE_SHIFT_GGUF_KEYS: frozenset = frozenset({
    "final_layer.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias",
})

# Format: comfy_gguf_key -> diffusers_key
# Verified against actual diffusers safetensors state dict
_MAP_BASIC = {
    "final_layer.linear.bias":               "proj_out.bias",
    "final_layer.linear.weight":             "proj_out.weight",
    "img_in.bias":                           "x_embedder.bias",
    "img_in.weight":                         "x_embedder.weight",
    # NOTE: LongCat has no vec_in/guidance_in (vec_in_dim=None, guidance_embed=False)
    # CRITICAL: LongCat uses 'time_embed' not 'time_text_embed' (FLUX uses time_text_embed)
    "time_in.in_layer.bias":                 "time_embed.timestep_embedder.linear_1.bias",
    "time_in.in_layer.weight":               "time_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.bias":                "time_embed.timestep_embedder.linear_2.bias",
    "time_in.out_layer.weight":              "time_embed.timestep_embedder.linear_2.weight",
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
    # fused QKV (img_attn.qkv) is split into to_q, to_k, to_v
    "img_attn.qkv.weight": [
        ("attn.to_q.weight", 0, 3),
        ("attn.to_k.weight", 1, 3),
        ("attn.to_v.weight", 2, 3),
    ],
    "img_attn.qkv.bias": [
        ("attn.to_q.bias", 0, 3),
        ("attn.to_k.bias", 1, 3),
        ("attn.to_v.bias", 2, 3),
    ],
    # fused txt QKV
    "txt_attn.qkv.weight": [
        ("attn.add_q_proj.weight", 0, 3),
        ("attn.add_k_proj.weight", 1, 3),
        ("attn.add_v_proj.weight", 2, 3),
    ],
    "txt_attn.qkv.bias": [
        ("attn.add_q_proj.bias", 0, 3),
        ("attn.add_k_proj.bias", 1, 3),
        ("attn.add_v_proj.bias", 2, 3),
    ],
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
    # fused QKV+MLP (linear1) is split into to_q, to_k, to_v, and proj_mlp
    "linear1.weight": [
        ("attn.to_q.weight", None),
        ("attn.to_k.weight", None),
        ("attn.to_v.weight", None),
        ("proj_mlp.weight", None),
    ],
    "linear1.bias": [
        ("attn.to_q.bias", None),
        ("attn.to_k.bias", None),
        ("attn.to_v.bias", None),
        ("proj_mlp.bias", None),
    ],
    "linear2.weight":         "proj_out.weight",
    "linear2.bias":           "proj_out.bias",
    "norm.query_norm.weight": "attn.norm_q.weight",
    "norm.key_norm.weight":   "attn.norm_k.weight",
}


# Build full forward map (comfy_name -> diffusers_name) for N double/single blocks
def _build_full_map(n_double: int = 10, n_single: int = 20) -> dict:
    m = dict(_MAP_BASIC)
    for i in range(n_double):
        for comfy_suffix, diffusers_suffix in _DOUBLE_BLOCK_MAP.items():
            if isinstance(diffusers_suffix, list):
                m[f"double_blocks.{i}.{comfy_suffix}"] = [
                    (f"transformer_blocks.{i}.{dst}", split_idx, total_splits)
                    for dst, split_idx, total_splits in diffusers_suffix
                ]
            else:
                m[f"double_blocks.{i}.{comfy_suffix}"] = f"transformer_blocks.{i}.{diffusers_suffix}"
    for i in range(n_single):
        for comfy_suffix, diffusers_suffix in _SINGLE_BLOCK_MAP.items():
            if isinstance(diffusers_suffix, list):
                m[f"single_blocks.{i}.{comfy_suffix}"] = [
                    (f"single_transformer_blocks.{i}.{dst}", split_info)
                    for dst, split_info in diffusers_suffix
                ]
            else:
                m[f"single_blocks.{i}.{comfy_suffix}"] = f"single_transformer_blocks.{i}.{diffusers_suffix}"
    return m

# Default map covers LongCat-Image (10/20). Edit model is handled dynamically in build_remap.
_FULL_MAP = _build_full_map(10, 20)


class LongCatKeyMap:
    """
    Translates ComfyUI LongCat GGUF tensor names (FLUX-style) back to
    Hugging Face Diffusers LongCatImageTransformer2DModel names.
    Handles both:
      - LongCat-Image      (10 double blocks, 20 single blocks)
      - LongCat-Image-Edit (19 double blocks, 38 single blocks)
    """
    NAME = "longcat"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        stripped = [k.replace("model.diffusion_model.", "") for k in gguf_keys]
        # Signature: FLUX-style img_in/txt_in with NO vec_in (LongCat has vec_in_dim=None)
        # and NO guidance_in (guidance_embed=False).
        # Both LongCat-Image (10 blocks) and LongCat-Image-Edit (19 blocks) match.
        # FLUX has 19 double blocks WITH vec_in — that's how we tell them apart.
        has_img_in  = any(k == "img_in.weight" for k in stripped)
        has_txt_in  = any(k == "txt_in.weight" for k in stripped)
        no_vec_in   = not any(k.startswith("vector_in.") for k in stripped)
        no_guidance = not any(k.startswith("guidance_in.") for k in stripped)
        has_double_blocks = any(k.startswith("double_blocks.0.") for k in stripped)
        return has_img_in and has_txt_in and no_vec_in and no_guidance and has_double_blocks

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        # Count blocks dynamically from actual file — handles both model variants
        stripped = [k.replace("model.diffusion_model.", "") for k in gguf_keys]
        n_double = sum(1 for k in stripped if k == f"img_in.weight")  # placeholder count start
        # Count by finding max double block index
        double_indices = set()
        single_indices = set()
        for k in stripped:
            if k.startswith("double_blocks."):
                try:
                    idx = int(k.split(".")[1])
                    double_indices.add(idx)
                except (IndexError, ValueError):
                    pass
            elif k.startswith("single_blocks."):
                try:
                    idx = int(k.split(".")[1])
                    single_indices.add(idx)
                except (IndexError, ValueError):
                    pass
        n_double = max(double_indices) + 1 if double_indices else 10
        n_single = max(single_indices) + 1 if single_indices else 20

        full_map = _build_full_map(n_double, n_single)
        mapping = {}
        for gguf_key in gguf_keys:
            clean = gguf_key.replace("model.diffusion_model.", "")
            if clean in full_map:
                target = full_map[clean]
                if isinstance(target, list):
                    # Handle fused QKV tensors which map to multiple diffusers keys
                    mapping[gguf_key] = []
                    for item in target:
                        if len(item) == 3:
                            mapping[gguf_key].append((item[0], (item[1], item[2])))
                        else:
                            mapping[gguf_key].append((item[0], item[1]))
                else:
                    mapping[gguf_key] = [(target, None)]
            else:
                mapping[gguf_key] = [(clean, None)]
        return mapping

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, orig_name: str) -> Any:
        import torch
        # norm_out: ComfyUI/BFL stores adaLN_modulation as [shift | scale],
        # but Diffusers AdaLayerNormContinuous.forward() reads [scale | shift].
        # Swap the two halves so scale and shift land in the correct slots.
        clean = orig_name.replace("model.diffusion_model.", "")
        if clean in _SWAP_SCALE_SHIFT_GGUF_KEYS:
            half = tensor.shape[0] // 2
            return torch.cat([tensor[half:], tensor[:half]], dim=0).contiguous()

        # Single block linear1 combines Q, K, V, and MLP.
        # Q, K, V are size `dim` each, MLP is size `4 * dim`. Total = 7 * dim.
        if "single_blocks" in orig_name and "linear1" in orig_name:
            dim = tensor.shape[0] // 7
            if "to_q" in diffusers_key:
                return tensor[0 : dim, ...]
            elif "to_k" in diffusers_key:
                return tensor[dim : dim*2, ...]
            elif "to_v" in diffusers_key:
                return tensor[dim*2 : dim*3, ...]
            elif "proj_mlp" in diffusers_key:
                return tensor[dim*3 : , ...]
        return tensor
