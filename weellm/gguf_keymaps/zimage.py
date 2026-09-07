"""
zimage.py -- GGUF key map for Z-Image format.

Detected by: keys starting with "context_refiner." or "noise_refiner.".
Uses imperative remapping because Z-Image keys are already in Diffusers-like
naming but with a few structural differences (fused QKV, renamed prefixes).
"""
from typing import Any, Dict, List


class ZImageKeyMap:
    NAME = "z-image"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return (
            any(k.startswith("context_refiner.") for k in gguf_keys)
            or any(k.startswith("noise_refiner.") for k in gguf_keys)
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for name in gguf_keys:
            entries = ZImageKeyMap._remap_one(name)
            remap[name] = entries
        return remap

    @staticmethod
    def _remap_one(name: str) -> List:
        # x_embedder / final_layer get a prefix swap
        if name.startswith("x_embedder."):
            return [(name.replace("x_embedder.", "all_x_embedder.2-1."), None)]
        if name.startswith("final_layer."):
            return [(name.replace("final_layer.", "all_final_layer.2-1."), None)]

        # Fused QKV → individual projections
        if ".attention." in name:
            if name.endswith(".attention.qkv.weight"):
                base = name.replace(".attention.qkv.weight", ".attention")
                return [
                    (f"{base}.to_q.weight", (0, 3)),
                    (f"{base}.to_k.weight", (1, 3)),
                    (f"{base}.to_v.weight", (2, 3)),
                ]
            if name.endswith(".attention.qkv.bias"):
                base = name.replace(".attention.qkv.bias", ".attention")
                return [
                    (f"{base}.to_q.bias", (0, 3)),
                    (f"{base}.to_k.bias", (1, 3)),
                    (f"{base}.to_v.bias", (2, 3)),
                ]
            if name.endswith(".attention.out.weight"):
                return [(name.replace(".attention.out.", ".attention.to_out.0."), None)]
            if name.endswith(".attention.out.bias"):
                return [(name.replace(".attention.out.", ".attention.to_out.0."), None)]
            if name.endswith(".attention.q_norm.weight"):
                return [(name.replace(".attention.q_norm.", ".attention.norm_q."), None)]
            if name.endswith(".attention.q_norm.bias"):
                return [(name.replace(".attention.q_norm.", ".attention.norm_q."), None)]
            if name.endswith(".attention.k_norm.weight"):
                return [(name.replace(".attention.k_norm.", ".attention.norm_k."), None)]
            if name.endswith(".attention.k_norm.bias"):
                return [(name.replace(".attention.k_norm.", ".attention.norm_k."), None)]

        # Pass-through: name is already Diffusers-compatible
        return [(name, None)]
