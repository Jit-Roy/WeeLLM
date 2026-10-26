"""
weellm/io/safetensors/keymaps/transformers/fibo2.py
Safetensors keymap for Bria Fibo 2.0 / Fibo 2.0 Edit Turbo checkpoint formats (e.g. asym_w4a8 ConvRot).

Detected by:
  - presence of "perceiver.layers." and "layers.0." in keys.
"""

from typing import Any, Dict, List
import torch

_DEQUANT_SUFFIXES = (
    ".svdq_l1",
    ".svdq_l2",
    ".comfy_quant",
    ".weight_scale",
    ".weight_codebook",
    ".weight_s_rel",
    ".weight_s_channel",
)


class Fibo2SafetensorsKeyMap:
    NAME = "fibo2-safetensors"

    @staticmethod
    def detect(keys: List[str]) -> bool:
        has_perceiver = any(k.startswith("perceiver.layers.") for k in keys)
        has_layers = any(k.startswith("layers.0.") for k in keys)
        return has_perceiver and has_layers

    @classmethod
    def build_remap(cls, keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for key in keys:
            # Exclude quantization side-tensors (.weight_scale, etc.) so they stay
            # as raw keys and can be consumed by the seeker dequantization pipeline.
            if any(key.endswith(s) for s in _DEQUANT_SUFFIXES):
                continue

            # 1. Self-attention fused QKV -> individual projections (to_q, to_k, to_v)
            if key.endswith(".attention.qkv.weight"):
                base = key[:-len(".attention.qkv.weight")] + ".attention"
                remap[key] = [
                    (f"{base}.to_q.weight", (0, 3)),
                    (f"{base}.to_k.weight", (1, 3)),
                    (f"{base}.to_v.weight", (2, 3)),
                ]
            elif key.endswith(".attention.qkv.bias"):
                base = key[:-len(".attention.qkv.bias")] + ".attention"
                remap[key] = [
                    (f"{base}.to_q.bias", (0, 3)),
                    (f"{base}.to_k.bias", (1, 3)),
                    (f"{base}.to_v.bias", (2, 3)),
                ]
            # 2. Attention projection output (out -> to_out.0)
            elif key.endswith(".attention.out.weight"):
                remap[key] = key.replace(".attention.out.weight", ".attention.to_out.0.weight")
            elif key.endswith(".attention.out.bias"):
                remap[key] = key.replace(".attention.out.bias", ".attention.to_out.0.bias")
            # 3. Attention Q/K norms (q_norm -> norm_q, k_norm -> norm_k)
            elif key.endswith(".attention.q_norm.weight"):
                remap[key] = key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
            elif key.endswith(".attention.k_norm.weight"):
                remap[key] = key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
            # 4. SwiGLU FFN (w13 -> w1, w3)
            elif key.endswith(".feed_forward.w13.weight"):
                base = key[:-len(".feed_forward.w13.weight")] + ".feed_forward"
                remap[key] = [
                    (f"{base}.w1.weight", (0, 2)),
                    (f"{base}.w3.weight", (1, 2)),
                ]
            # 5. Cross-attention (perceiver layers and main layers 6, 14, 22, 30, 38)
            elif key.endswith(".cross_attention.to_kv.weight"):
                base = key[:-len(".cross_attention.to_kv.weight")] + ".cross_attention"
                remap[key] = [
                    (f"{base}.to_k.weight", (0, 2)),
                    (f"{base}.to_v.weight", (1, 2)),
                ]
            elif key.endswith(".cross_attention.to_out.weight"):
                remap[key] = key.replace(".cross_attention.to_out.weight", ".cross_attention.to_out.0.weight")
            # 6. Perceiver self-attention (to_kv -> to_k, to_v; to_out -> to_out.0)
            elif key.endswith(".self_attention.to_kv.weight"):
                base = key[:-len(".self_attention.to_kv.weight")] + ".self_attention"
                remap[key] = [
                    (f"{base}.to_k.weight", (0, 2)),
                    (f"{base}.to_v.weight", (1, 2)),
                ]
            elif key.endswith(".self_attention.to_out.weight"):
                remap[key] = key.replace(".self_attention.to_out.weight", ".self_attention.to_out.0.weight")
            # 7. Perceiver FFN (w13 -> w1, w3)
            elif key.endswith(".ffn.w13.weight"):
                base = key[:-len(".ffn.w13.weight")] + ".ffn"
                remap[key] = [
                    (f"{base}.w1.weight", (0, 2)),
                    (f"{base}.w3.weight", (1, 2)),
                ]
            else:
                # Direct pass-through
                remap[key] = key
        return remap

    @staticmethod
    def postprocess_tensor(dst_key: str, tensor: torch.Tensor, src_key: str = "") -> torch.Tensor:
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
