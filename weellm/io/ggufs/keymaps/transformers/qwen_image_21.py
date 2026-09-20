"""
qwen_image_21.py -- GGUF key map for Qwen-Image-2.1 format.

Detected by: keys matching QwenImage21 naming (e.g. img_in.weight and transformer_blocks.0.attn...)
Uses imperative passthrough because the keys are already perfectly aligned with Diffusers.
"""
from typing import Any, Dict, List


class QwenImage21KeyMap:
    NAME = "qwen21"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        has_img_in = "img_in.weight" in gguf_keys
        has_transformer = any(k.startswith("transformer_blocks.") for k in gguf_keys)
        return has_img_in and has_transformer

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for name in gguf_keys:
            if "img_mlp.gate_up.weight" in name:
                base = name.replace("img_mlp.gate_up.weight", "img_mlp.")
                gate = base + "gate_layer.weight"
                proj = base + "proj.weight"
                remap[name] = [(gate, (0, 2)), (proj, (1, 2))]
            else:
                # 1:1 passthrough mapping for the rest
                remap[name] = [(name, None)]
        return remap
