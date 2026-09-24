"""GGUF key map for GLM text encoders used by CogView4."""

from typing import Any, Dict, List


class GLMKeyMap:
    NAME = "glm"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return (
            arch in ("glm", "glm4")
            or (
                "token_embd.weight" in gguf_keys
                and any(k.startswith("blk.") for k in gguf_keys)
                and "blk.0.ffn_up.weight" in gguf_keys
                and "blk.0.ffn_gate.weight" not in gguf_keys
            )
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap = {
            "token_embd.weight": [("embed_tokens.weight", None)],
            "output_norm.weight": [("norm.weight", None)],
            "output.weight": [("lm_head.weight", None)],
        }
        max_i = max(
            (int(parts[1]) for key in gguf_keys
             if (parts := key.split("."))[:1] == ["blk"] and len(parts) > 1 and parts[1].isdigit()),
            default=-1,
        )
        mapping = {
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
            "ffn_up": "mlp.gate_up_proj",
            "ffn_down": "mlp.down_proj",
        }
        for index in range(max_i + 1):
            for source, target in mapping.items():
                for suffix in ("weight", "bias"):
                    source_key = f"blk.{index}.{source}.{suffix}"
                    if source_key in gguf_keys:
                        remap[source_key] = [(f"layers.{index}.{target}.{suffix}", None)]
        return remap

