"""
weellm/io/ggufs/keymaps/transformers/fibo2.py
GGUF key map for Bria Fibo 2.0 and Fibo 2.0 Edit Turbo architectures.

Detected by:
  - presence of "perceiver.layers." and "layers.0." in GGUF keys.
"""

from typing import Any, Dict, List
import torch


class Fibo2GGUFKeyMap:
    NAME = "fibo2"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        has_perceiver = any(k.startswith("perceiver.layers.") for k in gguf_keys)
        has_layers = any(k.startswith("layers.0.") for k in gguf_keys)
        return has_perceiver and has_layers

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for name in gguf_keys:
            remap[name] = Fibo2GGUFKeyMap._remap_one(name)
        return remap

    @staticmethod
    def _remap_one(name: str) -> List:
        # 1. Self-attention fused QKV -> individual projections (to_q, to_k, to_v)
        if name.endswith(".attention.qkv.weight"):
            base = name[:-len(".attention.qkv.weight")] + ".attention"
            return [
                (f"{base}.to_q.weight", (0, 3)),
                (f"{base}.to_k.weight", (1, 3)),
                (f"{base}.to_v.weight", (2, 3)),
            ]
        if name.endswith(".attention.qkv.bias"):
            base = name[:-len(".attention.qkv.bias")] + ".attention"
            return [
                (f"{base}.to_q.bias", (0, 3)),
                (f"{base}.to_k.bias", (1, 3)),
                (f"{base}.to_v.bias", (2, 3)),
            ]

        # 2. Attention projection output (out -> to_out.0)
        if name.endswith(".attention.out.weight"):
            return [(name.replace(".attention.out.weight", ".attention.to_out.0.weight"), None)]
        if name.endswith(".attention.out.bias"):
            return [(name.replace(".attention.out.bias", ".attention.to_out.0.bias"), None)]

        # 3. Attention Q/K norms (q_norm -> norm_q, k_norm -> norm_k)
        if name.endswith(".attention.q_norm.weight"):
            return [(name.replace(".attention.q_norm.weight", ".attention.norm_q.weight"), None)]
        if name.endswith(".attention.k_norm.weight"):
            return [(name.replace(".attention.k_norm.weight", ".attention.norm_k.weight"), None)]

        # 4. SwiGLU FFN (w13 -> w1, w3)
        if name.endswith(".feed_forward.w13.weight"):
            base = name[:-len(".feed_forward.w13.weight")] + ".feed_forward"
            return [
                (f"{base}.w1.weight", (0, 2)),
                (f"{base}.w3.weight", (1, 2)),
            ]

        # 5. Cross-attention (perceiver layers and main layers 6, 14, 22, 30, 38)
        if name.endswith(".cross_attention.to_kv.weight"):
            base = name[:-len(".cross_attention.to_kv.weight")] + ".cross_attention"
            return [
                (f"{base}.to_k.weight", (0, 2)),
                (f"{base}.to_v.weight", (1, 2)),
            ]
        if name.endswith(".cross_attention.to_out.weight"):
            return [(name.replace(".cross_attention.to_out.weight", ".cross_attention.to_out.0.weight"), None)]

        # 6. Perceiver self-attention (to_kv -> to_k, to_v; to_out -> to_out.0)
        if name.endswith(".self_attention.to_kv.weight"):
            base = name[:-len(".self_attention.to_kv.weight")] + ".self_attention"
            return [
                (f"{base}.to_k.weight", (0, 2)),
                (f"{base}.to_v.weight", (1, 2)),
            ]
        if name.endswith(".self_attention.to_out.weight"):
            return [(name.replace(".self_attention.to_out.weight", ".self_attention.to_out.0.weight"), None)]

        # 7. Perceiver FFN (w13 -> w1, w3)
        if name.endswith(".ffn.w13.weight"):
            base = name[:-len(".ffn.w13.weight")] + ".ffn"
            return [
                (f"{base}.w1.weight", (0, 2)),
                (f"{base}.w3.weight", (1, 2)),
            ]

        # 8. Pass-through (all resident keys and already-matching layer tensors)
        return [(name, None)]

    @staticmethod
    def postprocess_tensor(dst_key: str, tensor: torch.Tensor, orig_name: str = "") -> torch.Tensor:
        """
        Unpad ComfyUI multiple-of-256 intermediate dimensions back to Diffusers native size:
        [9728, 3584] -> [9536, 3584] for w1 and w3, and [3584, 9728] -> [3584, 9536] for w2.
        """
        if (
            dst_key.endswith(".feed_forward.w1.weight")
            or dst_key.endswith(".feed_forward.w3.weight")
            or dst_key.endswith(".ffn.w1.weight")
            or dst_key.endswith(".ffn.w3.weight")
        ):
            if tensor.shape[0] == 9728:
                return tensor[:9536, :]
        elif (
            dst_key.endswith(".feed_forward.w2.weight")
            or dst_key.endswith(".ffn.w2.weight")
        ):
            if tensor.shape[-1] == 9728:
                return tensor[:, :9536]
        return tensor
