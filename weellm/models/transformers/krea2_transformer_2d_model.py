"""
krea2_transformer_2d_model.py -- Hook-based layer-streaming for Krea2Transformer2DModel.

Architecture (Krea-2-Turbo MMDiT):
  - text_fusion.layerwise_blocks (e.g. 2 blocks)
  - text_fusion.refiner_blocks (e.g. 2 blocks)
  - transformer_blocks (e.g. 28 blocks)

Strategy:
  - Resident on GPU: img_in, time_embed, time_mod_proj, text_fusion.projector,
                     txt_in, rotary_emb, final_layer (small, always needed)
  - Streamed: text_fusion.layerwise_blocks[i], text_fusion.refiner_blocks[i], transformer_blocks[i]
              (loaded just-in-time via hooks, evicted after forward pass)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

from torch.utils._python_dispatch import TorchDispatchMode
class VRAMTracker(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.max_vram = 0
        
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if torch.cuda.is_available():
            current_vram = torch.cuda.memory_allocated() / (1024**3)
            if current_vram > self.max_vram:
                self.max_vram = current_vram
                print(f"[Tracker] {func.__name__} | Peak VRAM: {current_vram:.3f} GB", flush=True)
        return out

_STREAMING_PREFIXES = ("text_fusion.layerwise_blocks.", "text_fusion.refiner_blocks.", "transformer_blocks.")


def _apply_krea2_runtime_patches() -> None:
    """Apply model-specific Krea2 runtime patches used to keep memory under control.

    These patches are intentionally kept in this model module so the generic
    pipeline remains architecture-agnostic. The Krea2 attention path is the most
    expensive part of the patch set; it manually slices query/key/value heads and
    dispatches them one head at a time, which avoids reusing the fused SDPA path
    and slows generation substantially compared with the native optimized path.
    """
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_krea2 import (
            Krea2AttnProcessor,
            Krea2FinalLayer,
            Krea2SwiGLU,
            Krea2TextFusionBlock,
            Krea2TransformerBlock,
            Krea2Transformer2DModel,
        )
        from diffusers.models.attention import FeedForward
    except ImportError:
        return

    if getattr(Krea2SwiGLU, "_weellm_patched", False):
        return

    global orig_swiglu_forward
    orig_swiglu_forward = Krea2SwiGLU.forward

    @torch.no_grad()
    def in_place_swiglu_forward(self, hidden_states):
        # Streaming hooks are now attached directly to self.gate, self.up, self.down
        # PyTorch will automatically load and evict their weights sequentially
        
        # 1. gate (loads gate.weight, evaluates, evicts gate.weight)
        gate_out = self.gate(hidden_states)
        
        # 2. in-place silu
        import torch.nn.functional as F
        F.silu(gate_out, inplace=True)
        
        # 3. up (loads up.weight, evaluates, evicts up.weight)
        up_out = self.up(hidden_states)
        
        # 4. in-place multiply
        gate_out.mul_(up_out)
        del up_out
        
        # 5. down (loads down.weight, evaluates, evicts down.weight)
        down_out = self.down(gate_out)
        return down_out

    @torch.no_grad()
    def patched_ff_forward(self, hidden_states, *args, **kwargs):
        if hidden_states.dtype == torch.float32 and hidden_states.shape[1] > 2000:
            import torch.nn.functional as F
            if hasattr(self.net[0], "proj"):
                chunks = []
                for chunk in hidden_states.split(128, dim=1):
                    chunks.append(orig_ff_forward(self, chunk, *args, **kwargs))
                return torch.cat(chunks, dim=1)
        return orig_ff_forward(self, hidden_states, *args, **kwargs)

    @torch.no_grad()
    def in_place_text_fusion_forward(self, hidden_states, attention_mask=None):
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(hidden_states=norm_hidden_states, attention_mask=attention_mask)
        hidden_states.add_(attn_output)
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states.add_(ff_output)
        return hidden_states

    @torch.no_grad()
    def in_place_transformer_block_forward(self, hidden_states, temb, image_rotary_emb=None, attention_mask=None):
        if hasattr(self, "attn_trigger"):
            self.attn_trigger(hidden_states)
            
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        norm_out = self.norm1(hidden_states)
        norm_out.mul_(1.0 + prescale).add_(preshift)

        attn_out = self.attn(
            norm_out,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_out.mul_(pregate)
        hidden_states.add_(attn_out)
        
        # Explicitly delete activation tensors to free VRAM immediately!
        del norm_out
        del attn_out
        
        if hasattr(self, "evict_attn"):
            self.evict_attn()
        
        if hasattr(self, "ff_trigger"):
            self.ff_trigger(hidden_states)

        norm_out2 = self.norm2(hidden_states)
        norm_out2.mul_(1.0 + postscale).add_(postshift)
        ff_out = self.ff(norm_out2)
        ff_out.mul_(postgate)
        hidden_states.add_(ff_out)
        
        del norm_out2
        del ff_out
        
        if hasattr(self, "evict_ff"):
            self.evict_ff()

        return hidden_states

    @torch.no_grad()
    def in_place_final_layer_forward(self, hidden_states, temb):
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        norm_out = self.norm(hidden_states)
        norm_out.mul_(1.0 + scale).add_(shift)
        return self.linear(norm_out)

    def chunked_attention_call(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        # Memory-efficient attention processing. 
        # Instead of materializing full Q (805MB) and Gate (805MB) tensors, we chunk
        # along the sequence dimension. K and V must be fully computed (268MB each).
        from diffusers.models.embeddings import apply_rotary_emb
        import math

        # 1. Compute full K and V (needed for cross-attention)
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        key = attn.norm_k(key)
        
        if image_rotary_emb is not None:
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            
        key = key.contiguous()
        value = value.contiguous()

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # 2. Output buffer
        hidden_states_out = torch.empty(
            batch_size,
            seq_len,
            attn.num_heads,
            attn.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        chunk_size = 16384  # Adjust chunk size to balance VRAM and speed. 16k is ~50MB chunks.
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            hs_chunk = hidden_states[:, i:end_i, :]

            # Compute Q chunk
            q_chunk = attn.to_q(hs_chunk).unflatten(-1, (attn.num_heads, attn.head_dim))
            q_chunk = attn.norm_q(q_chunk)
            
            # Slice rotary embeddings for this sequence chunk
            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                cos_chunk = cos[i:end_i, :]
                sin_chunk = sin[i:end_i, :]
                q_chunk = apply_rotary_emb(q_chunk, (cos_chunk, sin_chunk), sequence_dim=1)
            
            # Compute Gate chunk
            gate_chunk = attn.to_gate(hs_chunk)

            q_chunk = q_chunk.contiguous()

            # Dispatch attention!
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                out_chunk = dispatch_attention_fn(
                    q_chunk,
                    key,
                    value,
                    attn_mask=attention_mask,
                    enable_gqa=attn.num_heads != attn.num_kv_heads,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )

            # Multiply by gate and store
            out_chunk = out_chunk.flatten(2, 3)
            out_chunk = out_chunk * torch.sigmoid(gate_chunk)
            
            # Note: We keep it unprojected in the buffer because to_out[0] is Linear
            hidden_states_out[:, i:end_i, :] = out_chunk.unflatten(-1, (attn.num_heads, attn.head_dim))

            # Free chunk memory immediately
            del hs_chunk, q_chunk, gate_chunk, out_chunk
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        hidden_states_out = hidden_states_out.flatten(2, 3)
        return attn.to_out[0](hidden_states_out)

    orig_swiglu_forward = Krea2SwiGLU.forward
    Krea2SwiGLU.forward = in_place_swiglu_forward
    Krea2SwiGLU._weellm_patched = True

    orig_text_fusion_forward = Krea2TextFusionBlock.forward
    Krea2TextFusionBlock.forward = in_place_text_fusion_forward
    Krea2TextFusionBlock._weellm_patched = True

    orig_transformer_block_forward = Krea2TransformerBlock.forward
    Krea2TransformerBlock.forward = in_place_transformer_block_forward
    Krea2TransformerBlock._weellm_patched = True

    orig_final_layer_forward = Krea2FinalLayer.forward
    Krea2FinalLayer.forward = in_place_final_layer_forward
    Krea2FinalLayer._weellm_patched = True

    orig_attention_call = Krea2AttnProcessor.__call__
    Krea2AttnProcessor.__call__ = chunked_attention_call
    Krea2AttnProcessor._weellm_patched = True

    orig_ff_forward = FeedForward.forward
    FeedForward.forward = patched_ff_forward

    # Patch the top-level forward to cast float32 inputs to model dtype.
    # This is critical for bfloat16 runs: the diffusers pipeline feeds float32
    # latents and text embeddings, so without this cast all activations would
    # be float32 even though weights are bfloat16, doubling VRAM usage.
    orig_transformer_forward = Krea2Transformer2DModel.forward

    def patched_transformer_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Determine the model's param dtype from the first resident parameter.
        model_dtype = next(
            (p.dtype for p in self.parameters() if p.device.type != "meta"),
            None
        )
        if model_dtype is not None and model_dtype != torch.float32:
            hidden_states = hidden_states.to(dtype=model_dtype)
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        return orig_transformer_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)

    Krea2Transformer2DModel.forward = patched_transformer_forward
    Krea2Transformer2DModel._weellm_patched_fwd = True

    logger.info("      -> [WeeLLM] Krea2 runtime patches loaded in model module.")


class Krea2Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Wraps Krea2Transformer2DModel for memory-efficient streaming.
    Streams directly from original Hugging Face safetensors shards via live seek.
    """

    def _get_layer_keys(self, shard_name: str) -> List[str]:
        keys = []
        if shard_name.endswith(".attn_half"):
            base = shard_name[:-10]
            for k in self.seeker.weight_map:
                if k.startswith(base + ".norm1.") or k.startswith(base + ".attn.") or k.startswith(base + ".scale_shift_table"):
                    keys.append(k)
        elif shard_name.endswith(".ff_half"):
            base = shard_name[:-8]
            for k in self.seeker.weight_map:
                if k.startswith(base + ".norm2.") or k.startswith(base + ".ff."):
                    keys.append(k)
        else:
            for k in self.seeker.weight_map:
                if k.startswith(shard_name + "."):
                    keys.append(k)
        return keys

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        
        def add_block_halves(prefix, block):
            if not hasattr(block, "attn_trigger"):
                block.attn_trigger = nn.Identity()
                block.ff_trigger = nn.Identity()
                
                def evict_attn():
                    from weellm.memory import evict_module
                    from accelerate.utils import set_module_tensor_to_device
                    evict_module(block.norm1)
                    evict_module(block.attn)
                    set_module_tensor_to_device(block, "scale_shift_table", "meta")
                block.evict_attn = evict_attn
                
                def evict_ff():
                    from weellm.memory import evict_module
                    evict_module(block.norm2)
                    evict_module(block.ff)
                block.evict_ff = evict_ff
                
            order.append((f"{prefix}.attn_half", block.attn_trigger))
            order.append((f"{prefix}.ff_half", block.ff_trigger))

        # Text fusion blocks run first (stream as whole blocks because they are small and unpatched)
        for i, block in enumerate(self.model.text_fusion.layerwise_blocks):
            order.append((f"text_fusion.layerwise_blocks.{i}", block))
        for i, block in enumerate(self.model.text_fusion.refiner_blocks):
            order.append((f"text_fusion.refiner_blocks.{i}", block))
            
        # Then main joint blocks (stream in halves to fit in 4GB)
        for i, block in enumerate(self.model.transformer_blocks):
            add_block_halves(f"transformer_blocks.{i}", block)
            
        return order

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _STREAMING_PREFIXES) or "scale_shift_table" in k
        ]

    def _pre_hook(self, module: nn.Module, args):
        res = super()._pre_hook(module, args)
        if self.dtype == torch.float32 and getattr(self, "tracker", None) is None:
            self.tracker = VRAMTracker()
            self.tracker.__enter__()
        return res

    def _post_hook(self, module: nn.Module, args, output):
        res = super()._post_hook(module, args, output)
        if self.dtype == torch.float32:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return res

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "Krea2Transformer2DModelStreamer":
        from diffusers import Krea2Transformer2DModel

        transformer_dir = Path(transformer_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Krea-2-Turbo transformer weights ...")
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors.", len(seeker.weight_map))

        model = cls._load_model_on_meta(Krea2Transformer2DModel, transformer_dir, device, dtype, seeker)

        _apply_krea2_runtime_patches()

        logger.info("Step 3/3 -- Loading resident Krea-2-Turbo transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d layerwise, %d refiner, and %d joint transformer blocks for streaming.",
            len(model.text_fusion.layerwise_blocks),
            len(model.text_fusion.refiner_blocks),
            len(model.transformer_blocks),
        )
        logger.info("Krea2Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer

