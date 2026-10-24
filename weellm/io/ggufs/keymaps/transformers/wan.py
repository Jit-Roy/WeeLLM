"""
wan.py -- GGUF key map for Wan video DiT transformer (Wan2.1 / Wan2.2).

Detected by: general.architecture == "wan" in GGUF metadata.

The unsloth Wan GGUFs use native checkpoint naming which must be remapped
to diffusers WanTransformer3DModel attribute names.

Top-level prefix remaps:
  transformer_blocks.N.  ->  blocks.N.   (streaming blocks)
  time_embedding.        ->  condition_embedder.time_embedder.
  time_projection.       ->  condition_embedder.time_proj.
  text_embedding.        ->  condition_embedder.text_embedder.
  patch_embedding.       ->  patch_embedding.
  head.                  ->  proj_out. / scale_shift_table (see below)

Per-block sub-key remaps (inside blocks.N.*):
  self_attn.q.*          ->  attn1.to_q.*
  self_attn.k.*          ->  attn1.to_k.*
  self_attn.v.*          ->  attn1.to_v.*
  self_attn.o.*          ->  attn1.to_out.0.*
  self_attn.norm_q.*     ->  attn1.norm_q.*
  self_attn.norm_k.*     ->  attn1.norm_k.*
  cross_attn.q.*         ->  attn2.to_q.*
  cross_attn.k.*         ->  attn2.to_k.*
  cross_attn.v.*         ->  attn2.to_v.*
  cross_attn.o.*         ->  attn2.to_out.0.*
  cross_attn.norm_q.*    ->  attn2.norm_q.*
  cross_attn.norm_k.*    ->  attn2.norm_k.*
  ffn.0.*                ->  ffn.net.0.proj.*
  ffn.2.*                ->  ffn.net.2.*
  norm2.*                ->  norm2.*      (unchanged)
  modulation             ->  scale_shift_table

time_embedding sub-key remaps:
  time_embedding.0.*     ->  condition_embedder.time_embedder.linear_1.*
  time_embedding.2.*     ->  condition_embedder.time_embedder.linear_2.*

text_embedding sub-key remaps:
  text_embedding.0.*     ->  condition_embedder.text_embedder.linear_1.*
  text_embedding.2.*     ->  condition_embedder.text_embedder.linear_2.*

head sub-key remaps:
  head.modulation        ->  scale_shift_table
  head.head.*            ->  proj_out.*
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("weellm")


# ── Per-block sub-key renames ────────────────────────────────────────────────

# Maps (suffix_in_gguf_key -> suffix_in_diffusers_key) for patterns inside
# transformer_blocks.N.*
_BLOCK_SUB_KEY_MAP = [
    # Self-attention (attn1)
    (".self_attn.norm_q.",  ".attn1.norm_q."),
    (".self_attn.norm_k.",  ".attn1.norm_k."),
    (".self_attn.q.",       ".attn1.to_q."),
    (".self_attn.k.",       ".attn1.to_k."),
    (".self_attn.v.",       ".attn1.to_v."),
    (".self_attn.o.",       ".attn1.to_out.0."),
    # Cross-attention (attn2)
    (".cross_attn.norm_q.", ".attn2.norm_q."),
    (".cross_attn.norm_k.", ".attn2.norm_k."),
    (".cross_attn.q.",      ".attn2.to_q."),
    (".cross_attn.k.",      ".attn2.to_k."),
    (".cross_attn.v.",      ".attn2.to_v."),
    (".cross_attn.o.",      ".attn2.to_out.0."),
    # FFN
    (".ffn.0.",             ".ffn.net.0.proj."),
    (".ffn.2.",             ".ffn.net.2."),
    # Norm (ffn pre-norm): GGUF uses norm3, diffusers uses norm2
    (".norm3.",             ".norm2."),
    # Modulation → scale_shift_table (special: no dot-suffix after it)
    (".modulation",         ".scale_shift_table"),
]

# ── Top-level prefix remap (applied before block sub-key remap) ─────────────
# Note: transformer_blocks.N. -> blocks.N. is intentional — after this remap
# the key speaks diffusers naming for the WanTransformer3DModelStreamer.
_TOP_LEVEL_MAP = [
    ("transformer_blocks.",                        "blocks."),
    ("time_embedding.0.",                          "condition_embedder.time_embedder.linear_1."),
    ("time_embedding.2.",                          "condition_embedder.time_embedder.linear_2."),
    ("time_projection.0.",                         "condition_embedder.time_proj."),
    ("time_projection.1.",                         "condition_embedder.time_proj."),
    ("time_projection.",                           "condition_embedder.time_proj."),
    ("text_embedding.0.",                          "condition_embedder.text_embedder.linear_1."),
    ("text_embedding.2.",                          "condition_embedder.text_embedder.linear_2."),
    ("patch_embedding.",                           "patch_embedding."),
    ("head.head.",                                 "proj_out."),
    ("head.modulation",                            "scale_shift_table"),
]
# Sort longest first so more specific patterns match first
_TOP_LEVEL_MAP_SORTED = sorted(_TOP_LEVEL_MAP, key=lambda x: -len(x[0]))


def _apply_top_level_remap(gguf_key: str) -> str:
    for ckpt_prefix, diff_prefix in _TOP_LEVEL_MAP_SORTED:
        if gguf_key.startswith(ckpt_prefix):
            return diff_prefix + gguf_key[len(ckpt_prefix):]
    return gguf_key


def _apply_block_sub_key_remap(diffusers_key: str) -> str:
    """Apply per-block sub-key renaming to a key that already has the diffusers
    top-level prefix (i.e. starts with 'blocks.N.')."""
    for gguf_suffix, diff_suffix in _BLOCK_SUB_KEY_MAP:
        if gguf_suffix in diffusers_key:
            return diffusers_key.replace(gguf_suffix, diff_suffix, 1)
    return diffusers_key


class WanGGUFKeyMap:
    NAME = "wan"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return arch == "wan"

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for gguf_key in gguf_keys:
            diffusers_key = _apply_top_level_remap(gguf_key)
            diffusers_key = _apply_block_sub_key_remap(diffusers_key)
            remap[gguf_key] = [(diffusers_key, None)]
        return remap

    @staticmethod
    def patch_model_before_stream(model, config, device: str, dtype, seeker=None) -> None:
        """No structural patches needed for Wan."""
        pass
