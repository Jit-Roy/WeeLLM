"""
weellm/io/safetensors/keymaps/transformers/flux3.py
Safetensors keymap for FLUX.3 (FLUX 3 Image Klein / Klein-Base and its BFL variant exports).

Maps BFL reference DiT parameter names to Hugging Face Diffusers convention:
  - content_mode_blocks.<mode>.*  -> mode_blocks.<mode>.*
  - single_blocks.*              -> single_transformer_blocks.*
  - emb_in.*                     -> x_embedders.*
  - txt_in.*                     -> context_embedder.*
  - time_in.in_layer.*           -> timestep_embedder.linear_1.*
  - time_in.out_layer.*          -> timestep_embedder.linear_2.*
  - final_layer.image.adaLN_modulation.1.* -> norm_out.linear.*
  - final_layer.image.linear.*   -> proj_out.*

Also handles AdaLayerNorm output modulation chunk reordering:
  - BFL adaLN_modulation chunks: [shift, scale]
  - Diffusers Flux3AdaLayerNorm chunks: [scale, shift]
"""

from typing import Any, Dict, List
import torch

_DEQUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".comfy_quant",
    ".config",
    ".weight_codebook",
    ".weight_s_rel",
    ".weight_s_channel",
    ".svdq_l1",
    ".svdq_l2",
)


class Flux3SafetensorsKeyMap:
    NAME = "flux3-safetensors"

    @staticmethod
    def detect(keys: List[str]) -> bool:
        # Check for FLUX.3 distinct block structures: content_mode_blocks or single_blocks with emb_in
        return any(
            key.startswith(("content_mode_blocks.", "single_blocks.", "emb_in."))
            for key in keys
        )

    @staticmethod
    def _remap_key(key: str) -> str:
        replacements = (
            ("content_mode_blocks.", "mode_blocks."),
            ("single_blocks.", "single_transformer_blocks."),
            ("emb_in.", "x_embedders."),
            ("txt_in.", "context_embedder."),
            ("early_stream_modulations.", "early_stream_modulations."),
            ("single_stream_modulations.", "single_stream_modulations."),
            ("time_in.in_layer.", "timestep_embedder.linear_1."),
            ("time_in.out_layer.", "timestep_embedder.linear_2."),
            ("final_layer.image.adaLN_modulation.1.", "norm_out.linear."),
            ("final_layer.image.linear.", "proj_out."),
        )
        for source, target in replacements:
            if key.startswith(source):
                key = target + key[len(source):]
                break

        key = key.replace(".lin.", ".linear.")
        key = key.replace(".attn_out.", ".attn.to_out.0.")
        key = key.replace(".q_proj.", ".attn.to_q.")
        key = key.replace(".k_proj.", ".attn.to_k.")
        key = key.replace(".v_proj.", ".attn.to_v.")
        key = key.replace(".mlp_in.", ".ff.linear_in.")
        key = key.replace(".mlp_out.", ".ff.linear_out.")
        key = key.replace(".norm.query_norm.scale", ".attn.norm_q.weight")
        key = key.replace(".norm.key_norm.scale", ".attn.norm_k.weight")
        return key

    @classmethod
    def build_remap(cls, keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for key in keys:
            # Exclude quantization side-tensors (.weight_scale, etc.) so they stay
            # as raw keys and can be consumed by the seeker dequantization pipeline.
            if any(key.endswith(s) for s in _DEQUANT_SUFFIXES) or ".comfy_attention" in key:
                continue
            if ".qkv." in key:
                base_mapped = cls._remap_key(key)
                remap[key] = [
                    (base_mapped.replace(".qkv.", ".attn.to_q."), (0, 3)),
                    (base_mapped.replace(".qkv.", ".attn.to_k."), (1, 3)),
                    (base_mapped.replace(".qkv.", ".attn.to_v."), (2, 3)),
                ]
            else:
                remap[key] = cls._remap_key(key)
        return remap

    @staticmethod
    def postprocess_tensor(dst_key: str, tensor: torch.Tensor, src_key: str) -> torch.Tensor:
        # BFL adaLN_modulation chunks: [shift, scale] (each 3072 channels)
        # Diffusers Flux3AdaLayerNorm chunks: [scale, shift]
        # Swap the two halves so scale comes first and shift comes second.
        if (
            (dst_key == "norm_out.linear.weight" or dst_key.endswith(".norm_out.linear.weight"))
            and tensor.ndim == 2
            and tensor.shape[0] == 6144
        ):
            c0, c1 = tensor.chunk(2, dim=0)
            return torch.cat([c1, c0], dim=0)
        return tensor
