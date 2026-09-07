"""
llama.py -- GGUF key map for LLaMA / Qwen / Mistral / Gemma style models.

Detected by: keys starting with "blk." or "single_blk.".
"""
from typing import Any, Dict, List

_LLAMA_KEY_MAP = {
    "token_embd.weight":            "model.embed_tokens.weight",
    "output_norm.weight":           "model.norm.weight",
    "output.weight":                "lm_head.weight",
    "blk.{i}.attn_q.weight":        "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_q.bias":          "model.layers.{i}.self_attn.q_proj.bias",
    "blk.{i}.attn_k.weight":        "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_k.bias":          "model.layers.{i}.self_attn.k_proj.bias",
    "blk.{i}.attn_v.weight":        "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_v.bias":          "model.layers.{i}.self_attn.v_proj.bias",
    "blk.{i}.attn_output.weight":   "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.attn_output.bias":     "model.layers.{i}.self_attn.o_proj.bias",
    "blk.{i}.ffn_gate.weight":      "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight":        "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight":      "model.layers.{i}.mlp.down_proj.weight",
    "blk.{i}.attn_norm.weight":     "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight":      "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.attn_q_norm.weight":   "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight":   "model.layers.{i}.self_attn.k_norm.weight",
}


class LlamaKeyMap:
    NAME = "llama-gguf"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return any(k.startswith("blk.") or k.startswith("single_blk.") for k in gguf_keys)

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("blk", "single_blk") and len(parts) > 1:
                try:
                    max_i = max(max_i, int(parts[1]))
                except ValueError:
                    pass

        remap: Dict[str, Any] = {}
        for tmpl_src, tmpl_dst in _LLAMA_KEY_MAP.items():
            for i in range(max_i + 1):
                remap[tmpl_src.replace("{i}", str(i))] = [
                    (tmpl_dst.replace("{i}", str(i)), None)
                ]
        return remap
