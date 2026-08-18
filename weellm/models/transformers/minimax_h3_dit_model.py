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

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import default_dtype, clean_memory, report_memory

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


class MiniMaxH3DiTModelStreamer(BaseTransformerStreamer):
    """
    Wraps MiniMaxH3DiTModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        """Returns list of (diffusers_prefix, block_module) for streaming blocks."""
        order = []
        if hasattr(self.model, "transformer_blocks"):
            for i, block in enumerate(self.model.transformer_blocks):
                order.append((f"transformer_blocks.{i}", block))
        return order

    def _get_resident_ckpt_keys(self) -> List[str]:
        """Returns checkpoint keys (original MiniMax names) for non-streaming tensors."""
        return [
            k for k in self.seeker.weight_map
            if not k.startswith("blocks.")  # 'blocks.' is the ckpt prefix for transformer_blocks
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
        from weellm.memory import place_tensors

        remapped: Dict[str, torch.Tensor] = {}
        for ck, tensor in state_dict.items():
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
            # Rename out_proj → to_out.0
            elif ".attn.out_proj." in dk:
                dk = dk.replace(".attn.out_proj.", ".attn.to_out.0.")
                remapped[dk] = tensor
            # Rename norm keys: q_norm/k_norm → norm_q/norm_k
            elif ".attn.q_norm." in dk:
                dk = dk.replace(".attn.q_norm.", ".attn.norm_q.")
                remapped[dk] = tensor
            elif ".attn.k_norm." in dk:
                dk = dk.replace(".attn.k_norm.", ".attn.norm_k.")
                remapped[dk] = tensor
            # Rename mlp: fc1→ff.net.0.proj, fc2→ff.net.2
            elif ".mlp.fc1." in dk:
                gate, value = tensor.chunk(2, dim=0)
                remapped[dk.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat([value, gate], dim=0).contiguous()
            elif ".mlp.fc2." in dk:
                dk = dk.replace(".mlp.fc2.", ".ff.net.2.")
                remapped[dk] = tensor
            else:
                remapped[dk] = tensor

        from weellm.memory import place_tensors
        place_tensors(self.model, remapped, self.device, self.dtype, skip_errors=True)

    def _pre_hook(self, module: nn.Module, args):
        args = super()._pre_hook(module, args)
        
        from weellm.models.base_streamer import _SHARD_NAME_ATTR
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
    ) -> "MiniMaxH3DiTModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on MiniMax-H3 transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        # Step 2: Instantiate the model skeleton on meta device using from_config.
        # diffusers uses its own internal attribute names (proj_in, transformer_blocks, etc.)
        # We translate checkpoint keys → diffusers names when loading tensors.
        try:
            from diffusers import MiniMaxH3Transformer3DModel
            from accelerate import init_empty_weights
            from weellm.utils import default_dtype
            logger.info("  Instantiating MiniMaxH3Transformer3DModel on meta device ...")
            cfg = MiniMaxH3Transformer3DModel.load_config(str(transformer_dir))
            with init_empty_weights(), default_dtype(dtype):
                model = MiniMaxH3Transformer3DModel.from_config(cfg)
            model.eval()
        except Exception as e:
            logger.warning("diffusers instantiation failed (%s), falling back to Mock skeleton.", e)
            class MockMiniMaxH3DiTModel(nn.Module):
                def __init__(self, num_blocks: int = 50):
                    super().__init__()
                    self.blocks = nn.ModuleList([nn.Module() for _ in range(num_blocks)])
            model = MockMiniMaxH3DiTModel(num_blocks=50)
            model.eval()

        logger.info("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch, prefetch_device=prefetch_device)
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        if resident_ckpt_keys:
            # Load tensors using checkpoint key names, then apply prefix remapping
            # to translate to diffusers attribute names. Keys that still don't match
            # (e.g. fused qkv_proj vs split to_q/to_k/to_v) are skipped gracefully.
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device=device, dtype=dtype)
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

        block_count = len(streamer._get_shard_order())
        logger.info("Installed %d blocks for streaming.", block_count)
        logger.info("MiniMaxH3DiTModelStreamer ready. Mode: Live Seek from original shards")
        return streamer
