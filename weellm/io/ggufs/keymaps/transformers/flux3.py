"""
weellm/io/ggufs/keymaps/transformers/flux3.py
GGUF key map for FLUX.3 Image Klein.

Maps BFL/GGUF tensor names to Hugging Face Diffusers convention:
  - content_mode_blocks.<mode>.*  -> mode_blocks.<mode>.*
  - single_blocks.*              -> single_transformer_blocks.*
  - emb_in.*                     -> x_embedders.*
  - txt_in.*                     -> context_embedder.*
  - time_in.in_layer.*           -> timestep_embedder.linear_1.*
  - time_in.out_layer.*          -> timestep_embedder.linear_2.*
  - final_layer.image.adaLN_modulation.1.* -> norm_out.linear.*
  - final_layer.image.linear.*   -> proj_out.*
"""

from typing import Any, Dict, List
import torch


class Flux3GGUFKeyMap:
    NAME = "flux3"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return arch in ("flux3", "flux3-klein") or any(
            k.startswith(("content_mode_blocks.", "txt_mode_blocks.")) for k in gguf_keys
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
    def build_remap(cls, gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for key in gguf_keys:
            mapped = cls._remap_key(key)
            remap[key] = [(mapped, None)]
        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, orig_name: str) -> Any:
        # BFL adaLN_modulation chunks: [shift, scale] (each 3072 channels)
        # Diffusers Flux3AdaLayerNorm chunks: [scale, shift]
        # Swap the two halves so scale comes first and shift comes second.
        if (
            (diffusers_key == "norm_out.linear.weight" or diffusers_key.endswith(".norm_out.linear.weight"))
            and hasattr(tensor, "shape")
            and len(tensor.shape) == 2
            and tensor.shape[0] == 6144
        ):
            c0, c1 = tensor.chunk(2, dim=0)
            return torch.cat([c1, c0], dim=0)
        return tensor
