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
    Streams directly from original Hugging Face safetensors shards via live seek,
    OR from a GGUF quantized file (pass gguf_path to from_pretrained).

    Safetensors mode:
      - seeker.weight_map uses original MiniMax checkpoint keys (e.g. 'blocks.0.*')
      - _get_layer_keys() translates diffusers shard name -> checkpoint prefix
      - apply_state_dict() remaps checkpoint -> diffusers + splits fused QKV

    GGUF mode:
      - GGUFSeeker applies the MiniMaxH3KeyMap during __init__
      - seeker.weight_map already uses diffusers key names (e.g. 'transformer_blocks.0.*')
      - _get_layer_keys() uses diffusers prefix directly (no translation needed)
      - apply_state_dict() is a no-op remap since keys are already diffusers-named,
        but still handles the fused QKV split (encoded as slice_info in GGUFSeeker)
    """

    @property
    def _is_gguf(self) -> bool:
        """True when the backing seeker is a GGUFSeeker."""
        from weellm.gguf_seek import GGUFSeeker
        return isinstance(self.seeker, GGUFSeeker)

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        """Returns list of (diffusers_prefix, block_module) for streaming blocks."""
        order = []
        if hasattr(self.model, "transformer_blocks"):
            for i, block in enumerate(self.model.transformer_blocks):
                order.append((f"transformer_blocks.{i}", block))
        return order

    def _get_resident_ckpt_keys(self) -> List[str]:
        """Returns seeker weight-map keys for non-streaming (resident) tensors.

        In safetensors mode: filters checkpoint keys (those not starting with 'blocks.')
        In GGUF mode: filters diffusers keys (those not starting with 'transformer_blocks.')
        """
        if self._is_gguf:
            # GGUFSeeker weight_map is already in diffusers naming
            return [
                k for k in self.seeker.weight_map
                if not k.startswith("transformer_blocks.")
            ]
        return [
            k for k in self.seeker.weight_map
            if not k.startswith("blocks.")  # 'blocks.' is the ckpt prefix for transformer_blocks
        ]

    def _get_resident_keys(self) -> List[str]:
        """Alias expected by base class."""
        return self._get_resident_ckpt_keys()

    def _ckpt_shard_name(self, diffusers_shard_name: str) -> str:
        """Translate a diffusers shard name -> checkpoint shard name for safetensors seeker."""
        if diffusers_shard_name.startswith("transformer_blocks."):
            idx = diffusers_shard_name[len("transformer_blocks."):]
            return f"blocks.{idx}"
        return diffusers_shard_name

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        """Return seeker weight-map keys for this shard.

        In safetensors mode: translates diffusers shard name -> checkpoint prefix.
        In GGUF mode: GGUFSeeker weight_map already uses diffusers naming, so
                      we search directly by diffusers prefix.
        """
        if self._is_gguf:
            # shard_name is already a diffusers prefix (e.g. 'transformer_blocks.0')
            return [
                k for k in self.seeker.weight_map
                if k.startswith(shard_name + ".")
            ]
        ckpt_name = self._ckpt_shard_name(shard_name)
        return [
            k for k in self.seeker.weight_map
            if k.startswith(ckpt_name + ".")
        ]

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor], skip_errors: bool = False) -> None:
        """Remap keys -> diffusers names, then apply tensor transforms before placement.

        In safetensors mode: applies _remap_ckpt_key + all sub-key renames + QKV split.
        In GGUF mode: keys are already diffusers-named (GGUFSeeker applied keymap).
                      Only the mlp.fc1 gate/value swap needs to happen here.
        """
        from weellm.memory import place_tensors

        is_gguf = self._is_gguf
        remapped: Dict[str, torch.Tensor] = {}
        for ck, tensor in state_dict.items():
            # GGUF: keys already remapped; safetensors: apply checkpoint -> diffusers prefix remap
            dk = ck if is_gguf else _remap_ckpt_key(ck)

            if not is_gguf:
                # Split fused qkv_proj -> to_q/k/v (safetensors only; GGUF uses slice_info)
                if dk.endswith(".attn.qkv_proj.weight"):
                    prefix = dk[: -len("qkv_proj.weight")]
                    num_heads = self.model.config.num_attention_heads
                    head_dim = self.model.config.attention_head_dim
                    tensor = reorder_interleaved_qkv(tensor, num_heads, head_dim)
                    dim = tensor.shape[0] // 3
                    remapped[prefix + "to_q.weight"] = tensor[:dim].contiguous()
                    remapped[prefix + "to_k.weight"] = tensor[dim : 2 * dim].contiguous()
                    remapped[prefix + "to_v.weight"] = tensor[2 * dim :].contiguous()
                    continue
                elif dk.endswith(".attn.qkv_proj.bias"):
                    prefix = dk[: -len("qkv_proj.bias")]
                    dim = tensor.shape[0] // 3
                    remapped[prefix + "to_q.bias"] = tensor[:dim].contiguous()
                    remapped[prefix + "to_k.bias"] = tensor[dim : 2 * dim].contiguous()
                    remapped[prefix + "to_v.bias"] = tensor[2 * dim :].contiguous()
                    continue
                elif ".attn.out_proj." in dk:
                    dk = dk.replace(".attn.out_proj.", ".attn.to_out.0.")
                elif ".attn.q_norm." in dk:
                    dk = dk.replace(".attn.q_norm.", ".attn.norm_q.")
                elif ".attn.k_norm." in dk:
                    dk = dk.replace(".attn.k_norm.", ".attn.norm_k.")
                elif ".mlp.fc1." in dk:
                    gate, value = tensor.chunk(2, dim=0)
                    remapped[dk.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat([value, gate], dim=0).contiguous()
                    continue
                elif ".mlp.fc2." in dk:
                    dk = dk.replace(".mlp.fc2.", ".ff.net.2.")
            else:
                # GGUF: MiniMaxH3KeyMap renamed fc1->ff.net.0.proj; still need gate/value swap
                if ".ff.net.0.proj." in dk and tensor.dim() >= 1 and tensor.shape[0] > 1:
                    gate, value = tensor.chunk(2, dim=0)
                    remapped[dk] = torch.cat([value, gate], dim=0).contiguous()
                    continue

            remapped[dk] = tensor

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
        gguf_path: Optional[str] = None,
    ) -> "MiniMaxH3DiTModelStreamer":
        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on MiniMax-H3 transformer weights ...")
        if gguf_path is not None:
            from weellm.seeker import override_weights_path
            ctx = override_weights_path(gguf_path)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors in seeker weight_map.", len(seeker.weight_map))

        # Step 2: Instantiate the model skeleton on meta device using from_config.
        # diffusers uses its own internal attribute names (proj_in, transformer_blocks, etc.)
        # We translate checkpoint keys → diffusers names when loading tensors.
        from diffusers import MiniMaxH3Transformer3DModel
        from accelerate import init_empty_weights
        from weellm.utils import default_dtype
        logger.info("  Instantiating MiniMaxH3Transformer3DModel on meta device ...")
        cfg = MiniMaxH3Transformer3DModel.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = MiniMaxH3Transformer3DModel.from_config(cfg)
        model.eval()

        logger.info("Step 3/3 -- Loading resident transformer tensors to device=%s ...", device)
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch, prefetch_device=prefetch_device)
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        if resident_ckpt_keys:
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device=device, dtype=dtype)

            from weellm.gguf_seek import GGUFSeeker
            if isinstance(seeker, GGUFSeeker):
                # GGUF: keys in raw_sd are already in diffusers naming (GGUFSeeker applied keymap)
                # apply_state_dict will only apply the fc1 gate/value swap.
                remapped_sd = raw_sd
            else:
                # Safetensors: apply prefix remap + sub-key renames
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
