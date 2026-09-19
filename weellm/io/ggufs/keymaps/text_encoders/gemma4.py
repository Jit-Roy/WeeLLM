"""
gemma4.py -- GGUF key map for Gemma4 / LTX-2.5 text encoder GGUF format.

Unlike standard llama.cpp GGUFs (which use blk.* naming), the Gemma4-LTX-2.5
GGUF from elix3r already ships with diffusers-compatible key names:
  - model.embed_tokens.weight
  - model.layers.{i}.input_layernorm.weight
  - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  - model.layers.{i}.mlp.{gate,up,down}_proj.weight
  - model.layers.{i}.layer_scalar          (Gemma4-specific)
  - model.norm.weight
  - multi_modal_projector.embedding_projection.weight
  - audio_projector.embedding_projection.weight
  - vision_model.*
  - text_embedding_projection.*

Because the naming is already diffusers-compatible, build_remap() returns an
identity map (pass-through). The purpose of this class is purely:
  1. Detection  — claim the file so the correct arch name is logged.
  2. No-op remap — fall through cleanly to GGUFSeeker pass-through logic.

Detected by: presence of "multi_modal_projector.embedding_projection.weight"
             AND any "model.layers.*.layer_scalar" key.
"""

from typing import Any, Dict, List


class Gemma4KeyMap:
    NAME = "gemma4-ltx25-te"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        """
        Identify Gemma4 LTX-2.5 text-encoder GGUFs.

        Both conditions must hold:
          - Has the multimodal projector weight (unique to this merged GGUF format).
          - Has at least one layer_scalar key (Gemma4-specific per-layer scalar).
        """
        has_proj = "multi_modal_projector.embedding_projection.weight" in gguf_keys
        has_scalar = any(k.endswith(".layer_scalar") for k in gguf_keys)
        return has_proj and has_scalar

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        """
        Keys are already in diffusers format — return identity mapping for all
        of them so GGUFSeeker routes every key to itself unchanged.

        The Gemma4UnifiedForConditionalGenerationStreamer already handles
        the model.language_model.* vs model.* prefix difference at load time,
        so no prefix rewriting is needed here.
        """
        return {k: [(k, None)] for k in gguf_keys}
