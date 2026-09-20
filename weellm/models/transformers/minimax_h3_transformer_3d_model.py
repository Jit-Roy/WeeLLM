"""
minimax_h3_dit_model.py -- Hook-based layer-streaming for MiniMaxH3DiTModel.

Architecture:
  - 50 single-stream transformer blocks (dense)
  - 33B parameters

Strategy:
  - Resident on GPU: embedders, norm_out, proj_out
  - Streamed: transformer blocks (loaded just-in-time via hooks, evicted after forward pass)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
from weellm.io.seeker import get_seeker
from weellm.io.utils import default_dtype, clean_memory, report_memory

logger = logging.getLogger("weellm")

def reorder_interleaved_qkv(weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int) -> torch.Tensor:
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(f"fused qkv weight has {weight.shape[0]} rows, expected {expected_rows}")
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim, *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat(
        [
            tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:])
            for tensor in (query, key, value)
        ],
        dim=0,
    )

# The checkpoint uses original MiniMax key names; diffusers renames them.
# Top-level prefix remap (checkpoint → diffusers attribute names).
# NOTE: The attention qkv_proj (fused) vs to_q/to_k/to_v (split) mismatch
#       is handled by diffusers' own _convert_deprecated_attention_blocks,
#       so we only remap non-attention keys here.
_CKPT_PREFIX_REMAP: dict[str, str] = {
    "video_patch_proj.":          "proj_in.",
    "audio_patch_proj.":          "audio_proj_in.",
    "condition_proj.":            "context_embedder.",
    "blocks.":                    "transformer_blocks.",
    "token_refiner.blocks.":      "token_refiner.refiner_blocks.",
    "time_embedder.proj_in.":     "time_embedder.linear_1.",
    "time_embedder.proj_out.":    "time_embedder.linear_2.",
    "final_layer.norm.":          "norm_out.norm.",
    "final_layer.adaln_proj.":    "norm_out.",
    "final_layer.video_out.":     "proj_out.",
    "final_layer.audio_out.":     "audio_proj_out.",
}

# Diffusers streaming prefix (transformer blocks after remapping)
_STREAMING_PREFIXES = ("transformer_blocks.",)

# Checkpoint key prefix for streaming blocks (original MiniMax naming)
_CKPT_STREAMING_PREFIX = "blocks."


def _remap_ckpt_key(ckpt_key: str) -> str:
    """Apply top-level prefix remapping from checkpoint → diffusers attribute names."""
    for ckpt_prefix, diff_prefix in sorted(_CKPT_PREFIX_REMAP.items(), key=lambda x: -len(x[0])):
        if ckpt_key.startswith(ckpt_prefix):
            return diff_prefix + ckpt_key[len(ckpt_prefix):]
    return ckpt_key


class AdalNTableEmbedder(nn.Module):
    """Replaces time_embedder in pruned GGUF checkpoints.

    The pruned MiniMax-H3 GGUF replaces the heavy time_embedder MLP
    (256→5376→2688) with a compact lookup table of shape [1025, 8].
    During inference, the timestep t is mapped to a table index and
    linearly interpolated to produce an 8-dim embedding, which is then
    fed directly into each block's adaln_proj.linear (which has
    in_features=8 instead of 2688 in the pruned model).

    IMPORTANT — diffusers calling convention:
        diffusers' MiniMaxH3Transformer3DModel.forward pre-computes a
        sinusoidal embedding from the raw timestep and then calls
        ``temb = self.time_embedder(temb_sin)``.
        So our forward() receives the sinusoidal embedding, NOT raw t.
        We therefore intercept the raw timestep one level up via a
        model-level forward_pre_hook (installed in from_pretrained) and
        store it in ``self._raw_t``.

    Reference: ComfyUI comfy/ldm/minimax/model.py
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        # Keep table in float32 — using float16 causes NaN at early timesteps
        self.register_buffer("table", table.float(), persistent=True)
        # Set by the parent model's forward_pre_hook before each call
        self._raw_t: Optional[torch.Tensor] = None

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
        # VERIFIED via debug log: diffusers passes t ∈ [0, 1] directly.
        # This IS ComfyUI's internal t = 1 - sigma (not sigma, not sigma*1000).
        #   t=0.0 → noise (σ=1) → table[0]
        #   t=1.0 → clean (σ=0) → table[1024]
        # Matches ComfyUI source: pos = t.clamp(0,1) * (table.shape[0] - 1)
        t_norm = t.float().clamp(0.0, 1.0) * 1024.0  # [0..1024]
        # Index arithmetic on CPU (table buffer is on CPU)
        t_lo = t_norm.long().clamp(0, 1023).cpu()
        t_hi = (t_lo + 1).clamp(0, 1024)
        frac = (t_norm.cpu() - t_lo.float()).unsqueeze(-1)  # [M, 1]
        # torch.lerp between adjacent table entries (ComfyUI verified mechanism)
        basis = torch.lerp(self.table[t_lo], self.table[t_hi], frac)  # [M, 8]
        return basis.to(t.device)


class MiniMaxH3Transformer3DModelStreamer(BaseTransformerStreamer):
    """
    Wraps MiniMaxH3Transformer3DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        """Returns list of (diffusers_prefix, block_module) for streaming blocks."""
        order = []
        if hasattr(self.model, "transformer_blocks"):
            for i, block in enumerate(self.model.transformer_blocks):
                order.append((f"transformer_blocks.{i}", block))
        return order

    def _get_streaming_prefix(self) -> str:
        # Check if the GGUF uses diffusers naming or original naming
        if any(k.startswith("transformer_blocks.") for k in self.seeker.weight_map):
            return "transformer_blocks."
        return "blocks."

    def _get_resident_ckpt_keys(self) -> List[str]:
        """Returns checkpoint keys for non-streaming tensors."""
        prefix = self._get_streaming_prefix()
        return [
            k for k in self.seeker.weight_map
            if not k.startswith(prefix)
        ]

    def _get_resident_keys(self) -> List[str]:
        """Alias expected by base class — uses ckpt keys."""
        return self._get_resident_ckpt_keys()

    def _ckpt_shard_name(self, diffusers_shard_name: str) -> str:
        """Translate a diffusers shard name → checkpoint shard name for seeker lookups.

        The seeker's weight_map uses original MiniMax checkpoint keys (e.g. 'blocks.0'),
        but _get_shard_order() returns diffusers names ('transformer_blocks.0').
        This reversal is needed in _get_layer_keys so the streaming pre-hook can
        find the right tensors in the seeker.
        """
        prefix = self._get_streaming_prefix()
        if prefix == "transformer_blocks.":
            return diffusers_shard_name

        # Reverse the blocks.→transformer_blocks. mapping
        if diffusers_shard_name.startswith("transformer_blocks."):
            idx = diffusers_shard_name[len("transformer_blocks."):]
            return f"blocks.{idx}"
        return diffusers_shard_name

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        """Return seeker weight-map keys for this shard (using checkpoint naming)."""
        ckpt_name = self._ckpt_shard_name(shard_name)
        return [
            k for k in self.seeker.weight_map
            if k.startswith(ckpt_name + ".")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        """Remap checkpoint keys → diffusers names, then split fused qkv → to_q/k/v before placement."""
        from weellm.io.memory import place_tensors

        remapped: Dict[str, torch.Tensor] = {}
        # Iterate over keys and pop to free GPU memory immediately for modified tensors
        for ck in list(state_dict.keys()):
            tensor = state_dict.pop(ck)
            dk = _remap_ckpt_key(ck)  # prefix remap (blocks.X → transformer_blocks.X etc.)

            # Split fused qkv_proj weight/bias into separate to_q / to_k / to_v
            if dk.endswith(".attn.qkv_proj.weight"):
                prefix = dk[: -len("qkv_proj.weight")]
                # Raw weights are interleaved; we must reorder them to [q_all, k_all, v_all]
                num_heads = self.model.config.num_attention_heads
                head_dim = self.model.config.attention_head_dim
                tensor = reorder_interleaved_qkv(tensor, num_heads, head_dim)
                dim = tensor.shape[0] // 3
                remapped[prefix + "to_q.weight"] = tensor[:dim].contiguous()
                remapped[prefix + "to_k.weight"] = tensor[dim : 2 * dim].contiguous()
                remapped[prefix + "to_v.weight"] = tensor[2 * dim :].contiguous()
            elif dk.endswith(".attn.qkv_proj.bias"):
                prefix = dk[: -len("qkv_proj.bias")]
                dim = tensor.shape[0] // 3
                remapped[prefix + "to_q.bias"] = tensor[:dim].contiguous()
                remapped[prefix + "to_k.bias"] = tensor[dim : 2 * dim].contiguous()
                remapped[prefix + "to_v.bias"] = tensor[2 * dim :].contiguous()
                continue
            
            if ".attn.out_proj." in dk:
                dk = dk.replace(".attn.out_proj.", ".attn.to_out.0.")
            elif ".attn.q_norm." in dk:
                dk = dk.replace(".attn.q_norm.", ".attn.norm_q.")
            elif ".attn.k_norm." in dk:
                dk = dk.replace(".attn.k_norm.", ".attn.norm_k.")
            elif ".mlp.fc1." in dk:
                t_cpu = tensor.cpu()
                del tensor
                gate, value = t_cpu.chunk(2, dim=0)
                remapped[dk.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat([value, gate], dim=0)
                continue
            elif ".mlp.fc2." in dk:
                dk = dk.replace(".mlp.fc2.", ".ff.net.2.")
            else:
                # GGUF: MiniMaxH3KeyMap renamed fc1->ff.net.0.proj; still need gate/value swap
                if ".ff.net.0.proj." in dk and tensor.dim() >= 1 and tensor.shape[0] > 1:
                    # Swap on CPU to avoid VRAM fragmentation OOM (294 MiB torch.cat peak)
                    t_cpu = tensor.cpu()
                    del tensor
                    gate, value = t_cpu.chunk(2, dim=0)
                    remapped[dk] = torch.cat([value, gate], dim=0)
                    continue

            remapped[dk] = tensor

        from weellm.io.memory import place_tensors
        place_tensors(self.model, remapped, self.device, self.dtype, skip_errors=True)

    def _pre_hook(self, module: nn.Module, args):
        args = super()._pre_hook(module, args)
        
        from weellm.models.transformers.base_transformer_streamer import _SHARD_NAME_ATTR
        shard_name: str = getattr(module, _SHARD_NAME_ATTR)
        
        if hasattr(self, "lora_loader") and self.lora_loader is not None:
            self.lora_loader.apply_to_module(module, shard_name)
            
        return args

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        prefetch_device: Optional[str] = None,
        cache_to_ram: bool = False,
    ) -> "MiniMaxH3Transformer3DModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on MiniMax-H3 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        # Step 2: Instantiate the model skeleton on meta device using from_config.
        # diffusers uses its own internal attribute names (proj_in, transformer_blocks, etc.)
        # We translate checkpoint keys → diffusers names when loading tensors.
        from diffusers import MiniMaxH3Transformer3DModel
        from accelerate import init_empty_weights
        from weellm.io.utils import default_dtype
        logger.info("  Instantiating MiniMaxH3Transformer3DModel on meta device ...")
        cfg = MiniMaxH3Transformer3DModel.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = MiniMaxH3Transformer3DModel.from_config(cfg)
        model.eval()

        # ── Detect pruned GGUF (adaln_t_table present) ────────────────────────
        # In the pruned checkpoint, time_embedder is replaced by an 8-dim lookup
        # table, and every block's adaln_proj.linear has in_features=8 (not 2688).
        # We need to reinitialize those layers BEFORE loading GGUF weights so the
        # shapes match.
        _is_pruned = "adaln_t_table" in seeker.weight_map
        if _is_pruned:
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
            # We monkey-patch .forward on every module that applies the spurious SiLU.

            def _make_adaln_no_silu(lin, hidden_size):
                """Return a forward closure that skips SiLU before adaln_proj.linear."""
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

        logger.info("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch, prefetch_device=prefetch_device)
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        # Load adaln_t_table first (float32, on CPU) before other resident tensors
        if _is_pruned:
            table_tensor = seeker.get_tensors(["adaln_t_table"], device="cpu", dtype=torch.float32)
            adaln_table = table_tensor["adaln_t_table"]  # [1025, 8]
            embedder = AdalNTableEmbedder(adaln_table)
            model.time_embedder = embedder
            logger.info("  Installed AdalNTableEmbedder (table shape %s) — pruned mode.",
                        tuple(adaln_table.shape))

            # Register a model-level forward_pre_hook to capture raw timestep.
            # diffusers calls:  temb_sin = get_timestep_embedding(timestep, ...)
            #                   temb = self.time_embedder(temb_sin)
            # Our AdalNTableEmbedder.forward receives temb_sin (not raw t).
            # The hook fires before any of that, giving us raw timestep.
            _emb_ref = embedder  # captured by closure

            def _cache_raw_timestep(module, args, kwargs):
                t = kwargs.get("timestep")
                if t is None and len(args) >= 2:
                    t = args[1]   # positional: (hidden_states, timestep, ...)
                if t is not None:
                    _emb_ref._raw_t = t
                    if not getattr(_emb_ref, "_t_logged", False):
                        logger.info("  [adaln_t_debug] first timestep = %s  dtype=%s  values=%s",
                                    tuple(t.shape), t.dtype,
                                    t.flatten()[:4].tolist())
                        _emb_ref._t_logged = True
                return None  # do not modify args/kwargs

            model.register_forward_pre_hook(_cache_raw_timestep, with_kwargs=True)
            logger.info("  Registered raw-timestep pre-hook on transformer.")

        if resident_ckpt_keys:
            # Load tensors using checkpoint key names, then apply prefix remapping
            # to translate to diffusers attribute names. Keys that still don't match
            # (e.g. fused qkv_proj vs split to_q/to_k/to_v) are skipped gracefully.
            # We load to CPU first to prevent massive VRAM spikes when constructing the dict.
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device="cpu", dtype=dtype)
            remapped_sd = {_remap_ckpt_key(k): v for k, v in raw_sd.items()}
            # Handle diffusers token_refiner which splits qkv_proj, renames norms, and renames mlp
            new_remapped_sd = {}
            for k, v in remapped_sd.items():
                if "token_refiner" in k and "qkv_proj.weight" in k:
                    prefix = k.replace("qkv_proj.weight", "")
                    num_heads = model.config.num_attention_heads
                    head_dim = model.config.attention_head_dim
                    v = reorder_interleaved_qkv(v, num_heads, head_dim)
                    dim = v.shape[0] // 3
                    new_remapped_sd[prefix + "to_q.weight"] = v[:dim]
                    new_remapped_sd[prefix + "to_k.weight"] = v[dim:2*dim]
                    new_remapped_sd[prefix + "to_v.weight"] = v[2*dim:]
                elif "token_refiner" in k and "q_norm" in k:
                    new_remapped_sd[k.replace("q_norm", "norm_q")] = v
                elif "token_refiner" in k and "k_norm" in k:
                    new_remapped_sd[k.replace("k_norm", "norm_k")] = v
                elif "token_refiner" in k and "out_proj" in k:
                    new_remapped_sd[k.replace("out_proj", "to_out.0")] = v
                elif "token_refiner" in k and "mlp.fc1" in k:
                    gate, value = v.chunk(2, dim=0)
                    new_remapped_sd[k.replace("mlp.fc1", "ff.net.0.proj")] = torch.cat([value, gate], dim=0).contiguous()
                elif "token_refiner" in k and "mlp.fc2" in k:
                    new_remapped_sd[k.replace("mlp.fc2", "ff.net.2")] = v
                else:
                    new_remapped_sd[k] = v
            remapped_sd = new_remapped_sd

            model_keys = {n for n, _ in model.named_parameters()}
            skipped = [k for k in remapped_sd if k not in model_keys]
            if skipped:
                logger.info("  Skipping %d unmapped resident keys (e.g. fused attention): %s ...",
                            len(skipped), skipped[:3])
            streamer.apply_state_dict(remapped_sd, skip_errors=True)
            del remapped_sd

        clean_memory(device)
        report_memory("After resident load")

        # ── Post-load: materialise any params still on meta device ────────────
        from accelerate.utils.modeling import set_module_tensor_to_device
        streaming_prefixes = tuple(
            sname + "." for sname, _ in streamer._get_shard_order()
        )
        meta_names = [
            n for n, p in model.named_parameters()
            if p.device.type == "meta"
            and not any(n.startswith(pfx) for pfx in streaming_prefixes)
        ]
        if meta_names:
            logger.info(
                "  Initialising %d non-streaming meta params missing from GGUF: %s ...",
                len(meta_names), meta_names[:3],
            )
            with torch.no_grad():
                for pname in meta_names:
                    parts = pname.split(".")
                    mod = model
                    for part in parts[:-1]:
                        mod = getattr(mod, part)
                    param = getattr(mod, parts[-1])
                    new_t = torch.empty(param.shape, dtype=dtype, device=device)
                    if param.dim() >= 2:
                        torch.nn.init.xavier_uniform_(new_t)
                    else:
                        torch.nn.init.zeros_(new_t)
                    set_module_tensor_to_device(model, pname, device, value=new_t, dtype=dtype)
            logger.info("  Meta-param init done.")

        block_count = len(streamer._get_shard_order())
        logger.info("Installed %d blocks for streaming.", block_count)
        logger.info("MiniMaxH3Transformer3DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
