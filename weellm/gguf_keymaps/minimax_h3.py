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

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger("weellm")

class AdalNTableEmbedder(nn.Module):
    """Replaces time_embedder in pruned GGUF checkpoints.

    The pruned MiniMax-H3 GGUF replaces the heavy time_embedder MLP
    (256→5376→2688) with a compact lookup table of shape [1025, 8].
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        # Keep table in float32 — using float16 causes NaN at early timesteps
        self.register_buffer("table", table.float(), persistent=True)
        # Set by the parent model's forward_pre_hook before each call
        self._raw_t = None

    def forward(self, temb_sin: torch.Tensor) -> torch.Tensor:
        """temb_sin is the pre-computed sinusoidal embedding passed by diffusers.
        We ignore it and use ``self._raw_t`` (the original integer timestep
        in 0..1000) set by the model's forward_pre_hook."""
        if self._raw_t is None:
            raise RuntimeError(
                "AdalNTableEmbedder._raw_t is None — the model's "
                "forward_pre_hook may not have fired yet."
            )
        t = self._raw_t  # [M] in 0..1000  (diffusers scheduler convention)
        t_norm = t.float().clamp(0.0, 1.0) * 1024.0  # [0..1024]
        # Index arithmetic on CPU (table buffer is on CPU)
        t_lo = t_norm.long().clamp(0, 1023).cpu()
        t_hi = (t_lo + 1).clamp(0, 1024)
        frac = (t_norm.cpu() - t_lo.float()).unsqueeze(-1)  # [M, 1]
        # torch.lerp between adjacent table entries
        basis = torch.lerp(self.table[t_lo], self.table[t_hi], frac)  # [M, 8]
        return basis.to(t.device)

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
        # Match by architecture string or by MiniMax checkpoint key prefix.
        # 'wan' is the arch tag used by ComfyUI-packaged GGUFs of MiniMax-H3
        # (general.architecture=wan) — same key structure as minimax-h3.
        if arch in ("minimax-h3", "minimax_h3", "minimax", "wan"):
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

    @staticmethod
    def patch_model_before_stream(model: nn.Module, config, device: str, dtype: torch.dtype, seeker=None) -> None:
        """Apply any GGUF-specific structural patches."""
        if seeker is None:
            return

        # Check if this GGUF provides the pruned AdalN table.
        # It's an internal tensor we don't map to diffusers explicitly.
        # We can just probe seeker for it.
        try:
            table_tensor = seeker.get_tensors(["adaln_t_table"], device="cpu", dtype=torch.float32)
            if "adaln_t_table" in table_tensor:
                adaln_table = table_tensor["adaln_t_table"]  # [1025, 8]
                embedder = AdalNTableEmbedder(adaln_table)
                model.time_embedder = embedder
                logger.info("  Installed AdalNTableEmbedder (table shape %s) — pruned mode.",
                            tuple(adaln_table.shape))

                _emb_ref = embedder

                def _cache_raw_timestep(module, args, kwargs):
                    t = kwargs.get("timestep")
                    if t is None and len(args) >= 2:
                        t = args[1]
                    if t is not None:
                        _emb_ref._raw_t = t
                        if not getattr(_emb_ref, "_t_logged", False):
                            logger.info("  [adaln_t_debug] first timestep = %s  dtype=%s  values=%s",
                                        tuple(t.shape), t.dtype,
                                        t.flatten()[:4].tolist())
                            _emb_ref._t_logged = True
                    return None

                model.register_forward_pre_hook(_cache_raw_timestep, with_kwargs=True)
                logger.info("  Registered raw-timestep pre-hook on transformer.")

                logger.info("  Detected pruned GGUF (adaln_t_table present) — "
                            "reinitialising adaln_proj.linear in_features: 2688 → 8 on all blocks ...")
                
                # Reinitialise adaln_proj.linear to accept 8-dim input (matching the table output dim)
                # Must be done on CPU (not meta) so GGUF weights can be loaded in.
                for blk in model.transformer_blocks:
                    old_lin = blk.adaln_proj.linear
                    new_lin = nn.Linear(
                        in_features=8,
                        out_features=old_lin.out_features,
                        bias=old_lin.bias is not None,
                        device="cpu",
                        dtype=dtype,
                    )
                    blk.adaln_proj.linear = new_lin
                logger.info("  adaln_proj.linear reinitialised on %d blocks.", len(model.transformer_blocks))

                # norm_out.linear also uses temb in the same AdaLN pattern (from
                # final_layer.adaln_proj.linear in the GGUF) — patch it too.
                if hasattr(model, "norm_out") and hasattr(model.norm_out, "linear"):
                    old_norm_lin = model.norm_out.linear
                    model.norm_out.linear = nn.Linear(
                        in_features=8,
                        out_features=old_norm_lin.out_features,
                        bias=old_norm_lin.bias is not None,
                        device="cpu",
                        dtype=dtype,
                    )
                    logger.info("  norm_out.linear reinitialised to in_features=8.")

                # ── Patch adaln_proj and norm_out to skip SiLU ────────────────────
                # Diffusers hardcodes nn.functional.silu() before every adaln linear.
                # ComfyUI uses apply_silu=False for pruned GGUFs — the linear weights
                # were trained to receive the raw 8-dim table output directly.
                def _make_adaln_no_silu(lin, hidden_size):
                    def _fwd(temb: torch.Tensor):
                        out = lin(temb.to(lin.weight.dtype))
                        out = out.view(-1, 6 * hidden_size)
                        return out.chunk(6, dim=-1)
                    return _fwd

                for _blk in model.transformer_blocks:
                    _blk.adaln_proj.forward = _make_adaln_no_silu(
                        _blk.adaln_proj.linear, _blk.adaln_proj.hidden_size
                    )

                if hasattr(model, "norm_out") and hasattr(model.norm_out, "linear"):
                    _no_lin  = model.norm_out.linear
                    _no_norm = model.norm_out.norm

                    def _norm_out_no_silu(
                        hidden_states: torch.Tensor,
                        temb: torch.Tensor,
                        timestep_indices: torch.Tensor,
                        _lin=_no_lin,
                        _norm=_no_norm,
                    ) -> torch.Tensor:
                        shift, scale = _lin(temb.to(_lin.weight.dtype)).chunk(2, dim=-1)
                        hs = _norm(hidden_states)
                        return (
                            hs * (1.0 + scale.index_select(0, timestep_indices))
                            + shift.index_select(0, timestep_indices)
                        )

                    model.norm_out.forward = _norm_out_no_silu

                logger.info(
                    "  Patched adaln_proj (×%d) and norm_out to skip SiLU "
                    "(pruned GGUF: apply_silu=False).",
                    len(model.transformer_blocks),
                )

        except KeyError:
            # Not a pruned model, or doesn't have adaln_t_table
            pass