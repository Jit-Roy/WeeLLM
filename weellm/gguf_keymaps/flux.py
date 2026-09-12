"""
flux.py -- GGUF key map for Flux / Flux-Kontext format.

Detected by: keys starting with "double_blocks."
"""
from typing import Any, Dict, List

# Tensors in Flux GGUF format whose two output-half rows must be
# swapped before use with Diffusers.  The BFL→Diffusers conversion script
# applies swap_scale_shift() so Diffusers expects [shift | scale] order while
# the original GGUF stores [scale | shift].
_SWAP_SCALE_SHIFT_GGUF_KEYS: frozenset = frozenset({
    "final_layer.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias",
})

_FLUX_KEY_MAP = {
    # ── Double blocks ────────────────────────────────────────────────────────
    "double_blocks.{i}.img_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.to_q.weight",       0, 3),
        ("transformer_blocks.{i}.attn.to_k.weight",       1, 3),
        ("transformer_blocks.{i}.attn.to_v.weight",       2, 3),
    ],
    "double_blocks.{i}.img_attn.qkv.bias": [
        ("transformer_blocks.{i}.attn.to_q.bias",         0, 3),
        ("transformer_blocks.{i}.attn.to_k.bias",         1, 3),
        ("transformer_blocks.{i}.attn.to_v.bias",         2, 3),
    ],
    "double_blocks.{i}.txt_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.add_q_proj.weight", 0, 3),
        ("transformer_blocks.{i}.attn.add_k_proj.weight", 1, 3),
        ("transformer_blocks.{i}.attn.add_v_proj.weight", 2, 3),
    ],
    "double_blocks.{i}.txt_attn.qkv.bias": [
        ("transformer_blocks.{i}.attn.add_q_proj.bias",   0, 3),
        ("transformer_blocks.{i}.attn.add_k_proj.bias",   1, 3),
        ("transformer_blocks.{i}.attn.add_v_proj.bias",   2, 3),
    ],
    "double_blocks.{i}.img_attn.proj.weight":            "transformer_blocks.{i}.attn.to_out.0.weight",
    "double_blocks.{i}.img_attn.proj.bias":              "transformer_blocks.{i}.attn.to_out.0.bias",
    "double_blocks.{i}.txt_attn.proj.weight":            "transformer_blocks.{i}.attn.to_add_out.weight",
    "double_blocks.{i}.txt_attn.proj.bias":              "transformer_blocks.{i}.attn.to_add_out.bias",
    "double_blocks.{i}.img_mlp.0.weight":                "transformer_blocks.{i}.ff.linear_in.weight",
    "double_blocks.{i}.img_mlp.0.bias":                  "transformer_blocks.{i}.ff.linear_in.bias",
    "double_blocks.{i}.img_mlp.2.weight":                "transformer_blocks.{i}.ff.linear_out.weight",
    "double_blocks.{i}.img_mlp.2.bias":                  "transformer_blocks.{i}.ff.linear_out.bias",
    "double_blocks.{i}.txt_mlp.0.weight":                "transformer_blocks.{i}.ff_context.linear_in.weight",
    "double_blocks.{i}.txt_mlp.0.bias":                  "transformer_blocks.{i}.ff_context.linear_in.bias",
    "double_blocks.{i}.txt_mlp.2.weight":                "transformer_blocks.{i}.ff_context.linear_out.weight",
    "double_blocks.{i}.txt_mlp.2.bias":                  "transformer_blocks.{i}.ff_context.linear_out.bias",
    "double_blocks.{i}.img_mod.lin.weight":              "transformer_blocks.{i}.norm1.linear.weight",
    "double_blocks.{i}.img_mod.lin.bias":                "transformer_blocks.{i}.norm1.linear.bias",
    "double_blocks.{i}.txt_mod.lin.weight":              "transformer_blocks.{i}.norm1_context.linear.weight",
    "double_blocks.{i}.txt_mod.lin.bias":                "transformer_blocks.{i}.norm1_context.linear.bias",
    "double_blocks.{i}.img_attn.norm.key_norm.scale":   "transformer_blocks.{i}.attn.norm_k.weight",
    "double_blocks.{i}.img_attn.norm.query_norm.scale": "transformer_blocks.{i}.attn.norm_q.weight",
    "double_blocks.{i}.txt_attn.norm.key_norm.scale":   "transformer_blocks.{i}.attn.norm_added_k.weight",
    "double_blocks.{i}.txt_attn.norm.query_norm.scale": "transformer_blocks.{i}.attn.norm_added_q.weight",

    # ── Single blocks ────────────────────────────────────────────────────────
    "single_blocks.{i}.linear1.weight":       "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight",
    "single_blocks.{i}.linear1.bias":         "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.bias",
    "single_blocks.{i}.linear2.weight":       "single_transformer_blocks.{i}.attn.to_out.weight",
    "single_blocks.{i}.linear2.bias":         "single_transformer_blocks.{i}.attn.to_out.bias",
    "single_blocks.{i}.modulation.lin.weight":"single_transformer_blocks.{i}.norm.linear.weight",
    "single_blocks.{i}.modulation.lin.bias":  "single_transformer_blocks.{i}.norm.linear.bias",
    "single_blocks.{i}.norm.key_norm.scale":  "single_transformer_blocks.{i}.attn.norm_k.weight",
    "single_blocks.{i}.norm.query_norm.scale":"single_transformer_blocks.{i}.attn.norm_q.weight",

    # ── Top-level ────────────────────────────────────────────────────────────
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight":     "single_stream_modulation.linear.weight",
    "time_in.in_layer.weight":                 "time_text_embed.timestep_embedder.linear_1.weight",
    "time_in.in_layer.bias":                   "time_text_embed.timestep_embedder.linear_1.bias",
    "time_in.out_layer.weight":                "time_text_embed.timestep_embedder.linear_2.weight",
    "time_in.out_layer.bias":                  "time_text_embed.timestep_embedder.linear_2.bias",
    "guidance_in.in_layer.weight":             "time_text_embed.guidance_embedder.linear_1.weight",
    "guidance_in.in_layer.bias":               "time_text_embed.guidance_embedder.linear_1.bias",
    "guidance_in.out_layer.weight":            "time_text_embed.guidance_embedder.linear_2.weight",
    "guidance_in.out_layer.bias":              "time_text_embed.guidance_embedder.linear_2.bias",
    "vector_in.in_layer.weight":               "time_text_embed.text_embedder.linear_1.weight",
    "vector_in.in_layer.bias":                 "time_text_embed.text_embedder.linear_1.bias",
    "vector_in.out_layer.weight":              "time_text_embed.text_embedder.linear_2.weight",
    "vector_in.out_layer.bias":                "time_text_embed.text_embedder.linear_2.bias",
    "txt_in.weight":                           "context_embedder.weight",
    "txt_in.bias":                             "context_embedder.bias",
    "img_in.weight":                           "x_embedder.weight",
    "img_in.bias":                             "x_embedder.bias",
    "final_layer.linear.weight":               "proj_out.weight",
    "final_layer.linear.bias":                 "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight":   "norm_out.linear.weight",
    "final_layer.adaLN_modulation.1.bias":     "norm_out.linear.bias",
}


class FluxKeyMap:
    NAME = "flux"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return any(k.startswith("double_blocks.") for k in gguf_keys)

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("double_blocks", "single_blocks") and len(parts) > 1:
                try:
                    max_i = max(max_i, int(parts[1]))
                except ValueError:
                    pass

        remap: Dict[str, Any] = {}
        for tmpl_src, tmpl_dst in _FLUX_KEY_MAP.items():
            for i in range(max_i + 1):
                src = tmpl_src.replace("{i}", str(i))
                if isinstance(tmpl_dst, list):
                    remap[src] = [
                        (dst.replace("{i}", str(i)), (split_idx, total_splits))
                        for dst, split_idx, total_splits in tmpl_dst
                    ]
                else:
                    remap[src] = [(tmpl_dst.replace("{i}", str(i)), None)]
        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, orig_name: str) -> Any:
        import torch
        # Flux GGUF stores norm_out weights in [scale | shift] order;
        # Diffusers expects [shift | scale].
        if orig_name in _SWAP_SCALE_SHIFT_GGUF_KEYS:
            half = tensor.shape[0] // 2
            return torch.cat([tensor[half:], tensor[:half]], dim=0).contiguous()
        return tensor