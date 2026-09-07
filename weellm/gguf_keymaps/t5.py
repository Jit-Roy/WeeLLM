"""
t5.py -- GGUF key map for T5 encoder (city96 convention, arch=t5encoder).

Detected by: arch == "t5encoder" OR keys starting with "enc.blk.".
"""
from typing import Any, Dict, List

_T5_KEY_MAP = {
    # Top-level
    "token_embd.weight":              "shared.weight",
    "enc.output_norm.weight":         "encoder.final_layer_norm.weight",
    # Per-block attention
    "enc.blk.{i}.attn_q.weight":     "encoder.block.{i}.layer.0.SelfAttention.q.weight",
    "enc.blk.{i}.attn_k.weight":     "encoder.block.{i}.layer.0.SelfAttention.k.weight",
    "enc.blk.{i}.attn_v.weight":     "encoder.block.{i}.layer.0.SelfAttention.v.weight",
    "enc.blk.{i}.attn_o.weight":     "encoder.block.{i}.layer.0.SelfAttention.o.weight",
    "enc.blk.{i}.attn_rel_b.weight": "encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight",
    "enc.blk.{i}.attn_norm.weight":  "encoder.block.{i}.layer.0.layer_norm.weight",
    # Per-block FFN (T5 v1.1 gated: gate=wi_0, up=wi_1)
    "enc.blk.{i}.ffn_gate.weight":   "encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight",
    "enc.blk.{i}.ffn_up.weight":     "encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight",
    "enc.blk.{i}.ffn_down.weight":   "encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
    "enc.blk.{i}.ffn_norm.weight":   "encoder.block.{i}.layer.1.layer_norm.weight",
}


class T5KeyMap:
    NAME = "t5-encoder"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return arch == "t5encoder" or any(k.startswith("enc.blk.") for k in gguf_keys)

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "enc" and parts[1] == "blk":
                try:
                    max_i = max(max_i, int(parts[2]))
                except ValueError:
                    pass

        remap: Dict[str, Any] = {}
        for tmpl_src, tmpl_dst in _T5_KEY_MAP.items():
            if "{i}" not in tmpl_src:
                remap[tmpl_src] = [(tmpl_dst, None)]
            else:
                for i in range(max_i + 1):
                    remap[tmpl_src.replace("{i}", str(i))] = [
                        (tmpl_dst.replace("{i}", str(i)), None)
                    ]
        return remap
