"""
minimax_h3.py -- GGUF key map for MiniMax H3 DiT transformer.

Detected by: keys starting with "blocks.0." (MiniMax original checkpoint naming)
             OR general.architecture == "minimax-h3" in GGUF metadata.

The unsloth GGUF uses the original MiniMax checkpoint key names, which match
the _CKPT_PREFIX_REMAP in minimax_h3_dit_model.py. This keymap translates
those GGUF keys directly into diffusers attribute names so the GGUFSeeker's
weight_map speaks the same language as the streamer's _get_layer_keys().

Key name mappings derived from comparing:
  - Original checkpoint keys (from model.safetensors.index.json)
  - GGUF tensor names (from the unsloth MiniMax-H3-GGUF repo)

Top-level prefix remaps (matching _CKPT_PREFIX_REMAP):
  video_patch_proj.  -> proj_in.
  audio_patch_proj.  -> audio_proj_in.
  condition_proj.    -> context_embedder.
  blocks.            -> transformer_blocks.
  token_refiner.blocks. -> token_refiner.refiner_blocks.
  time_embedder.proj_in.  -> time_embedder.linear_1.
  time_embedder.proj_out. -> time_embedder.linear_2.
  final_layer.norm.       -> norm_out.norm.
  final_layer.adaln_proj. -> norm_out.
  final_layer.video_out.  -> proj_out.
  final_layer.audio_out.  -> audio_proj_out.

Per-block sub-key remaps (inside transformer_blocks.N.*):
  attn.qkv_proj.weight  -> [attn.to_q.weight, attn.to_k.weight, attn.to_v.weight]  (split, interleaved)
  attn.qkv_proj.bias    -> [attn.to_q.bias,   attn.to_k.bias,   attn.to_v.bias]
  attn.out_proj.*       -> attn.to_out.0.*
  attn.q_norm.*         -> attn.norm_q.*
  attn.k_norm.*         -> attn.norm_k.*
  mlp.fc1.*             -> ff.net.0.proj.*  (with gate/value chunk swap)
  mlp.fc2.*             -> ff.net.2.*
"""

from typing import Any, Dict, List

# ── Top-level prefix map: checkpoint name -> diffusers name ──────────────────
_TOP_LEVEL_MAP = {
    "video_patch_proj.":       "proj_in.",
    "audio_patch_proj.":       "audio_proj_in.",
    "condition_proj.":         "context_embedder.",
    "token_refiner.blocks.":   "token_refiner.refiner_blocks.",
    "blocks.":                 "transformer_blocks.",
    "time_embedder.proj_in.":  "time_embedder.linear_1.",
    "time_embedder.proj_out.": "time_embedder.linear_2.",
    "final_layer.norm.":       "norm_out.norm.",
    "final_layer.adaln_proj.": "norm_out.",
    "final_layer.video_out.":  "proj_out.",
    "final_layer.audio_out.":  "audio_proj_out.",
}

# Sort by descending key length so more specific patterns match first
_TOP_LEVEL_MAP_SORTED = sorted(_TOP_LEVEL_MAP.items(), key=lambda x: -len(x[0]))


def _apply_top_level_remap(ckpt_key: str) -> str:
    for ckpt_prefix, diff_prefix in _TOP_LEVEL_MAP_SORTED:
        if ckpt_key.startswith(ckpt_prefix):
            return diff_prefix + ckpt_key[len(ckpt_prefix):]
    return ckpt_key


def _build_block_entries(gguf_key: str, diffusers_key: str) -> list:
    """
    Given a GGUF key (already top-level remapped to diffusers naming),
    expand any sub-key renames (attn, mlp, etc.) and return a list of
    (diffusers_name, slice_info) tuples.

    slice_info is None or (split_idx, total_splits).
    """
    # attn.qkv_proj.weight -> to_q.weight, to_k.weight, to_v.weight (interleaved split)
    if diffusers_key.endswith(".attn.qkv_proj.weight"):
        prefix = diffusers_key[: -len("qkv_proj.weight")]
        return [
            (prefix + "to_q.weight", (0, 3)),
            (prefix + "to_k.weight", (1, 3)),
            (prefix + "to_v.weight", (2, 3)),
        ]
    if diffusers_key.endswith(".attn.qkv_proj.bias"):
        prefix = diffusers_key[: -len("qkv_proj.bias")]
        return [
            (prefix + "to_q.bias", (0, 3)),
            (prefix + "to_k.bias", (1, 3)),
            (prefix + "to_v.bias", (2, 3)),
        ]

    # attn.out_proj.* -> attn.to_out.0.*
    if ".attn.out_proj." in diffusers_key:
        return [(diffusers_key.replace(".attn.out_proj.", ".attn.to_out.0."), None)]

    # attn.q_norm.* -> attn.norm_q.*
    if ".attn.q_norm." in diffusers_key:
        return [(diffusers_key.replace(".attn.q_norm.", ".attn.norm_q."), None)]

    # attn.k_norm.* -> attn.norm_k.*
    if ".attn.k_norm." in diffusers_key:
        return [(diffusers_key.replace(".attn.k_norm.", ".attn.norm_k."), None)]

    # mlp.fc1.* -> ff.net.0.proj.*
    # Note: the actual gate/value swap happens in apply_state_dict, not here.
    # The keymap just does the name rename; the streamer's apply_state_dict
    # handles the torch chunk() + reorder.
    if ".mlp.fc1." in diffusers_key:
        return [(diffusers_key.replace(".mlp.fc1.", ".ff.net.0.proj."), None)]

    # mlp.fc2.* -> ff.net.2.*
    if ".mlp.fc2." in diffusers_key:
        return [(diffusers_key.replace(".mlp.fc2.", ".ff.net.2."), None)]

    # token_refiner sub-key renames
    if "token_refiner" in diffusers_key:
        dk = diffusers_key
        if ".attn.out_proj." in dk:
            dk = dk.replace(".attn.out_proj.", ".attn.to_out.0.")
        if ".attn.q_norm." in dk:
            dk = dk.replace(".attn.q_norm.", ".attn.norm_q.")
        if ".attn.k_norm." in dk:
            dk = dk.replace(".attn.k_norm.", ".attn.norm_k.")
        if ".mlp.fc1." in dk:
            dk = dk.replace(".mlp.fc1.", ".ff.net.0.proj.")
        if ".mlp.fc2." in dk:
            dk = dk.replace(".mlp.fc2.", ".ff.net.2.")
        if "qkv_proj.weight" in dk:
            prefix = dk[: dk.index("qkv_proj.weight")]
            return [
                (prefix + "to_q.weight", (0, 3)),
                (prefix + "to_k.weight", (1, 3)),
                (prefix + "to_v.weight", (2, 3)),
            ]
        if "qkv_proj.bias" in dk:
            prefix = dk[: dk.index("qkv_proj.bias")]
            return [
                (prefix + "to_q.bias", (0, 3)),
                (prefix + "to_k.bias", (1, 3)),
                (prefix + "to_v.bias", (2, 3)),
            ]
        if dk != diffusers_key:
            return [(dk, None)]

    return [(diffusers_key, None)]


class MiniMaxH3KeyMap:
    NAME = "minimax-h3"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # Match by architecture string or by MiniMax checkpoint key prefix
        if arch in ("minimax-h3", "minimax_h3", "minimax"):
            return True
        # Original checkpoint keys use "blocks.N." prefix
        return any(k.startswith("blocks.") for k in gguf_keys)

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for gguf_key in gguf_keys:
            # Step 1: top-level prefix remap (checkpoint -> diffusers naming)
            diffusers_key = _apply_top_level_remap(gguf_key)
            # Step 2: per-block sub-key renames (attn, mlp, etc.)
            entries = _build_block_entries(gguf_key, diffusers_key)
            remap[gguf_key] = entries
        return remap
