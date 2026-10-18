"""GGUF key map for Stable Diffusion 3 and SD3.5 transformers."""

from typing import Any, Dict, List


_SWAP_SCALE_SHIFT_GGUF_KEYS = frozenset({
    "final_layer.adaLN_modulation.1.bias",
    "final_layer.adaLN_modulation.1.weight",
    "joint_blocks.23.context_block.adaLN_modulation.1.bias",
    "joint_blocks.23.context_block.adaLN_modulation.1.weight",
})


class SD3GGUFKeyMap:
    NAME = "sd3"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        has_joint_blocks = any(k.startswith("joint_blocks.") for k in gguf_keys)
        has_image_stream = any(".x_block." in k for k in gguf_keys)
        has_context_stream = any(".context_block." in k for k in gguf_keys)
        has_fused_qkv = any(k.endswith(".attn.qkv.weight") for k in gguf_keys)
        return has_joint_blocks and has_image_stream and has_context_stream and has_fused_qkv

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}

        top_level = {
            "context_embedder.bias": "context_embedder.bias",
            "context_embedder.weight": "context_embedder.weight",
            "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
            "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
            "final_layer.linear.bias": "proj_out.bias",
            "final_layer.linear.weight": "proj_out.weight",
            "pos_embed": "pos_embed.pos_embed",
            "t_embedder.mlp.0.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "t_embedder.mlp.0.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "t_embedder.mlp.2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
            "t_embedder.mlp.2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
            "x_embedder.proj.bias": "pos_embed.proj.bias",
            "x_embedder.proj.weight": "pos_embed.proj.weight",
            "y_embedder.mlp.0.bias": "time_text_embed.text_embedder.linear_1.bias",
            "y_embedder.mlp.0.weight": "time_text_embed.text_embedder.linear_1.weight",
            "y_embedder.mlp.2.bias": "time_text_embed.text_embedder.linear_2.bias",
            "y_embedder.mlp.2.weight": "time_text_embed.text_embedder.linear_2.weight",
        }
        for source, target in top_level.items():
            remap[source] = [(target, None)]

        max_block = -1
        for key in gguf_keys:
            parts = key.split(".")
            if len(parts) > 1 and parts[0] == "joint_blocks":
                try:
                    max_block = max(max_block, int(parts[1]))
                except ValueError:
                    pass

        for index in range(max_block + 1):
            source_prefix = f"joint_blocks.{index}"
            target_prefix = f"transformer_blocks.{index}"

            direct = {
                f"{source_prefix}.x_block.adaLN_modulation.1.bias":
                    f"{target_prefix}.norm1.linear.bias",
                f"{source_prefix}.x_block.adaLN_modulation.1.weight":
                    f"{target_prefix}.norm1.linear.weight",
                f"{source_prefix}.x_block.attn.proj.bias":
                    f"{target_prefix}.attn.to_out.0.bias",
                f"{source_prefix}.x_block.attn.proj.weight":
                    f"{target_prefix}.attn.to_out.0.weight",
                f"{source_prefix}.x_block.mlp.fc1.bias":
                    f"{target_prefix}.ff.net.0.proj.bias",
                f"{source_prefix}.x_block.mlp.fc1.weight":
                    f"{target_prefix}.ff.net.0.proj.weight",
                f"{source_prefix}.x_block.mlp.fc2.bias":
                    f"{target_prefix}.ff.net.2.bias",
                f"{source_prefix}.x_block.mlp.fc2.weight":
                    f"{target_prefix}.ff.net.2.weight",
                f"{source_prefix}.context_block.adaLN_modulation.1.bias":
                    f"{target_prefix}.norm1_context.linear.bias",
                f"{source_prefix}.context_block.adaLN_modulation.1.weight":
                    f"{target_prefix}.norm1_context.linear.weight",
                f"{source_prefix}.context_block.attn.proj.bias":
                    f"{target_prefix}.attn.to_add_out.bias",
                f"{source_prefix}.context_block.attn.proj.weight":
                    f"{target_prefix}.attn.to_add_out.weight",
                f"{source_prefix}.context_block.mlp.fc1.bias":
                    f"{target_prefix}.ff_context.net.0.proj.bias",
                f"{source_prefix}.context_block.mlp.fc1.weight":
                    f"{target_prefix}.ff_context.net.0.proj.weight",
                f"{source_prefix}.context_block.mlp.fc2.bias":
                    f"{target_prefix}.ff_context.net.2.bias",
                f"{source_prefix}.context_block.mlp.fc2.weight":
                    f"{target_prefix}.ff_context.net.2.weight",
            }
            for source, target in direct.items():
                remap[source] = [(target, None)]

            for stream in ("x_block", "context_block"):
                source_attn_prefix = f"{source_prefix}.{stream}.attn"
                target_attn_prefix = f"{target_prefix}.attn"
                norm_prefix = "norm_added" if stream == "context_block" else "norm"
                remap[f"{source_attn_prefix}.ln_q.weight"] = [
                    (f"{target_attn_prefix}.{norm_prefix}_q.weight", None)
                ]
                remap[f"{source_attn_prefix}.ln_k.weight"] = [
                    (f"{target_attn_prefix}.{norm_prefix}_k.weight", None)
                ]

            for stream, target_prefix_part in (
                ("x_block", "attn"),
                ("context_block", "attn"),
            ):
                source = f"{source_prefix}.{stream}.attn.qkv"
                target_names = (
                    ("to_q", "to_k", "to_v")
                    if stream == "x_block"
                    else ("add_q_proj", "add_k_proj", "add_v_proj")
                )
                for suffix in ("weight", "bias"):
                    remap[f"{source}.{suffix}"] = [
                        (f"{target_prefix}.{target_prefix_part}.{name}.{suffix}", (split, 3))
                        for split, name in enumerate(target_names)
                    ]

            dual_source = f"{source_prefix}.x_block.attn2"
            dual_target = f"{target_prefix}.attn2"
            for suffix in ("weight", "bias"):
                remap[f"{dual_source}.qkv.{suffix}"] = [
                    (f"{dual_target}.to_{name}.{suffix}", (split, 3))
                    for split, name in enumerate(("q", "k", "v"))
                ]
            remap[f"{dual_source}.proj.weight"] = [(f"{dual_target}.to_out.0.weight", None)]
            remap[f"{dual_source}.proj.bias"] = [(f"{dual_target}.to_out.0.bias", None)]
            remap[f"{dual_source}.ln_q.weight"] = [(f"{dual_target}.norm_q.weight", None)]
            remap[f"{dual_source}.ln_k.weight"] = [(f"{dual_target}.norm_k.weight", None)]

        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, orig_name: str) -> Any:
        if orig_name in _SWAP_SCALE_SHIFT_GGUF_KEYS:
            import torch

            half = tensor.shape[0] // 2
            return torch.cat([tensor[half:], tensor[:half]], dim=0).contiguous()
        return tensor
