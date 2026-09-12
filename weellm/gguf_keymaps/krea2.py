"""
krea2.py -- GGUF key map for Krea2 / MiniMax H3 format.

Detected by: keys starting with "txtfusion." or "blocks.0.attn.qknorm.".
"""
from typing import Any, Dict, List

_KREA2_KEY_MAP = {
    # Time embedding
    "tmlp.0.weight":  "time_embed.linear_1.weight",
    "tmlp.0.bias":    "time_embed.linear_1.bias",
    "tmlp.2.weight":  "time_embed.linear_2.weight",
    "tmlp.2.bias":    "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias":   "time_mod_proj.bias",

    # Text input projection
    "txtmlp.0.scale":  "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias":   "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias":   "txt_in.linear_2.bias",

    # Image in/out
    "first.weight": "img_in.weight",
    "first.bias":   "img_in.bias",

    # Final layer
    "last.norm.scale":       "final_layer.norm.weight",
    "last.modulation.lin":   "final_layer.scale_shift_table",
    "last.linear.weight":    "final_layer.linear.weight",
    "last.linear.bias":      "final_layer.linear.bias",

    # Text fusion projector
    "txtfusion.projector.weight": "text_fusion.projector.weight",

    # ── transformer_blocks ────────────────────────────────────────────────────
    "blocks.{i}.attn.qknorm.knorm.scale": "transformer_blocks.{i}.attn.norm_k.weight",
    "blocks.{i}.attn.qknorm.qnorm.scale": "transformer_blocks.{i}.attn.norm_q.weight",
    "blocks.{i}.attn.gate.weight":         "transformer_blocks.{i}.attn.to_gate.weight",
    "blocks.{i}.attn.wk.weight":           "transformer_blocks.{i}.attn.to_k.weight",
    "blocks.{i}.attn.wo.weight":           "transformer_blocks.{i}.attn.to_out.0.weight",
    "blocks.{i}.attn.wq.weight":           "transformer_blocks.{i}.attn.to_q.weight",
    "blocks.{i}.attn.wv.weight":           "transformer_blocks.{i}.attn.to_v.weight",
    "blocks.{i}.mlp.down.weight":          "transformer_blocks.{i}.ff.down.weight",
    "blocks.{i}.mlp.gate.weight":          "transformer_blocks.{i}.ff.gate.weight",
    "blocks.{i}.mlp.up.weight":            "transformer_blocks.{i}.ff.up.weight",
    "blocks.{i}.prenorm.scale":            "transformer_blocks.{i}.norm1.weight",
    "blocks.{i}.postnorm.scale":           "transformer_blocks.{i}.norm2.weight",
    "blocks.{i}.mod.lin":                  "transformer_blocks.{i}.scale_shift_table",

    # ── txtfusion layerwise_blocks ────────────────────────────────────────────
    "txtfusion.layerwise_blocks.{i}.attn.qknorm.knorm.scale": "text_fusion.layerwise_blocks.{i}.attn.norm_k.weight",
    "txtfusion.layerwise_blocks.{i}.attn.qknorm.qnorm.scale": "text_fusion.layerwise_blocks.{i}.attn.norm_q.weight",
    "txtfusion.layerwise_blocks.{i}.attn.gate.weight":         "text_fusion.layerwise_blocks.{i}.attn.to_gate.weight",
    "txtfusion.layerwise_blocks.{i}.attn.wk.weight":           "text_fusion.layerwise_blocks.{i}.attn.to_k.weight",
    "txtfusion.layerwise_blocks.{i}.attn.wo.weight":           "text_fusion.layerwise_blocks.{i}.attn.to_out.0.weight",
    "txtfusion.layerwise_blocks.{i}.attn.wq.weight":           "text_fusion.layerwise_blocks.{i}.attn.to_q.weight",
    "txtfusion.layerwise_blocks.{i}.attn.wv.weight":           "text_fusion.layerwise_blocks.{i}.attn.to_v.weight",
    "txtfusion.layerwise_blocks.{i}.mlp.down.weight":          "text_fusion.layerwise_blocks.{i}.ff.down.weight",
    "txtfusion.layerwise_blocks.{i}.mlp.gate.weight":          "text_fusion.layerwise_blocks.{i}.ff.gate.weight",
    "txtfusion.layerwise_blocks.{i}.mlp.up.weight":            "text_fusion.layerwise_blocks.{i}.ff.up.weight",
    "txtfusion.layerwise_blocks.{i}.prenorm.scale":            "text_fusion.layerwise_blocks.{i}.norm1.weight",
    "txtfusion.layerwise_blocks.{i}.postnorm.scale":           "text_fusion.layerwise_blocks.{i}.norm2.weight",

    # ── txtfusion refiner_blocks ───────────────────────────────────────────────
    "txtfusion.refiner_blocks.{i}.attn.qknorm.knorm.scale": "text_fusion.refiner_blocks.{i}.attn.norm_k.weight",
    "txtfusion.refiner_blocks.{i}.attn.qknorm.qnorm.scale": "text_fusion.refiner_blocks.{i}.attn.norm_q.weight",
    "txtfusion.refiner_blocks.{i}.attn.gate.weight":         "text_fusion.refiner_blocks.{i}.attn.to_gate.weight",
    "txtfusion.refiner_blocks.{i}.attn.wk.weight":           "text_fusion.refiner_blocks.{i}.attn.to_k.weight",
    "txtfusion.refiner_blocks.{i}.attn.wo.weight":           "text_fusion.refiner_blocks.{i}.attn.to_out.0.weight",
    "txtfusion.refiner_blocks.{i}.attn.wq.weight":           "text_fusion.refiner_blocks.{i}.attn.to_q.weight",
    "txtfusion.refiner_blocks.{i}.attn.wv.weight":           "text_fusion.refiner_blocks.{i}.attn.to_v.weight",
    "txtfusion.refiner_blocks.{i}.mlp.down.weight":          "text_fusion.refiner_blocks.{i}.ff.down.weight",
    "txtfusion.refiner_blocks.{i}.mlp.gate.weight":          "text_fusion.refiner_blocks.{i}.ff.gate.weight",
    "txtfusion.refiner_blocks.{i}.mlp.up.weight":            "text_fusion.refiner_blocks.{i}.ff.up.weight",
    "txtfusion.refiner_blocks.{i}.prenorm.scale":            "text_fusion.refiner_blocks.{i}.norm1.weight",
    "txtfusion.refiner_blocks.{i}.postnorm.scale":           "text_fusion.refiner_blocks.{i}.norm2.weight",
}


class Krea2KeyMap:
    NAME = "krea2-minimax"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return (
            any(k.startswith("txtfusion.") for k in gguf_keys)
            or any(k.startswith("blocks.0.attn.qknorm.") for k in gguf_keys)
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("blocks", "txtfusion") and len(parts) > 1:
                try:
                    idx = int(parts[2]) if parts[1].endswith("_blocks") else int(parts[1])
                    max_i = max(max_i, idx)
                except (ValueError, IndexError):
                    pass

        remap: Dict[str, Any] = {}
        for tmpl_src, tmpl_dst in _KREA2_KEY_MAP.items():
            if "{i}" not in tmpl_src:
                remap[tmpl_src] = [(tmpl_dst, None)]
            else:
                for i in range(max_i + 1):
                    remap[tmpl_src.replace("{i}", str(i))] = [
                        (tmpl_dst.replace("{i}", str(i)), None)
                    ]
        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, orig_name: str) -> Any:
        import torch
        # Krea2 scale_shift_table is stored flat in GGUF but must be [6, dim]
        if diffusers_key.endswith("scale_shift_table") and tensor.dim() == 1:
            return tensor.reshape(6, -1)
        return tensor