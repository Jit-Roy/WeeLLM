"""
weellm/io/safetensors/keymaps/transformers/krea2.py
Safetensors keymap for the gonzalomo / ComfyUI Krea-2 checkpoint format.

ComfyUI Krea-2 checkpoints (e.g. gonzalomoKrea2_v40.safetensors) use the
namespace:  model.diffusion_model.<rest>
where <rest> matches the GGUF Krea2 key convention (blocks.{i}.attn.wq.weight,
txtfusion.layerwise_blocks.{i}.*, etc.).

This keymap strips the outer "model.diffusion_model." prefix so that the
resulting keys are identical to what the GGUF Krea2KeyMap produces — and
therefore identical to diffusers convention.

Detection:  any key starts with "model.diffusion_model."
"""
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Shared key-template table — intentionally identical to the GGUF krea2.py map.
# This is the single source of truth for the Krea-2 naming convention.
# ---------------------------------------------------------------------------
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

    # ── transformer_blocks ──────────────────────────────────────────────────
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

    # ── txtfusion layerwise_blocks ──────────────────────────────────────────
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

    # ── txtfusion refiner_blocks ────────────────────────────────────────────
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

_OUTER_PREFIX = "model.diffusion_model."

# Keys that are consumed by dequantization — they appear in weight_map but must
# NOT be remapped to diffusers names (the seeker drops them after dequant).
_DEQUANT_SUFFIXES = (".svdq_l1", ".svdq_l2", ".comfy_quant", ".weight_scale",
                     ".weight_codebook", ".weight_s_rel", ".weight_s_channel")


class Krea2SafetensorsKeyMap:
    NAME = "krea2-comfyui-safetensors"

    @staticmethod
    def detect(keys: List[str]) -> bool:
        # Require Krea2-specific markers under model.diffusion_model.*
        # "model.diffusion_model.*" alone is NOT sufficient — SDXL and SD1.5
        # checkpoints also use that prefix (input_blocks.*, output_blocks.*, etc.).
        # We check for txtfusion.* or blocks.*.attn.wq.* which are unique to Krea2.
        has_txtfusion = any(k.startswith("model.diffusion_model.txtfusion.") for k in keys)
        has_krea2_attn = any(
            "model.diffusion_model.blocks." in k and ".attn.wq." in k
            for k in keys
        )
        return has_txtfusion or has_krea2_attn

    @staticmethod
    def build_remap(keys: List[str]) -> Dict[str, str]:
        """
        Build a flat {gonzalomo_key: diffusers_key} dict for every key in *keys*.
        Keys with no mapping entry pass through unchanged (after prefix strip).
        Dequant side-tensors are excluded — they are handled by comfy_dequant.
        """
        # Count block indices so we can expand {i} templates
        max_i: Dict[str, int] = {"blocks": 0, "txtfusion": 0}
        idx_re = re.compile(r"^model\.diffusion_model\.(blocks|txtfusion\.\w+_blocks)\.(\d+)\.")
        for k in keys:
            m = idx_re.match(k)
            if m:
                grp = "txtfusion" if m.group(1).startswith("txtfusion") else "blocks"
                max_i[grp] = max(max_i[grp], int(m.group(2)))

        # Build expanded template → diffusers dict
        template_remap: Dict[str, str] = {}
        for tmpl_src, tmpl_dst in _KREA2_KEY_MAP.items():
            if "{i}" not in tmpl_src:
                template_remap[tmpl_src] = tmpl_dst
            else:
                grp = "txtfusion" if tmpl_src.startswith("txtfusion") else "blocks"
                for i in range(max_i[grp] + 1):
                    template_remap[tmpl_src.replace("{i}", str(i))] = tmpl_dst.replace("{i}", str(i))

        # Map gonzalomo full keys → diffusers keys
        remap: Dict[str, str] = {}
        for k in keys:
            if not k.startswith(_OUTER_PREFIX):
                continue
            if any(k.endswith(s) for s in _DEQUANT_SUFFIXES):
                continue  # consumed by dequant pipeline
            inner = k[len(_OUTER_PREFIX):]
            dst = template_remap.get(inner, inner)  # fall back to inner key if no mapping
            remap[k] = dst

        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, src_key: str) -> Any:
        """Reshape scale_shift_table from flat [N] → [N//6144, 6144]."""
        if diffusers_key.endswith("scale_shift_table") and tensor.dim() == 1:
            return tensor.reshape(-1, 6144)
        return tensor
