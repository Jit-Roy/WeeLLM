"""
weellm.pipelines.video.minimax_h3_modular_pipeline
==================================================
WeeLLM adapter for the MiniMax-H3 Modular Pipeline.

Follows the same pattern as inference.py:
  - Standard WeeBasePipeline.from_pretrained() loads all components
    (VAE, transformer, text encoder via GGUF override_weights_path)
  - Post-load patches fix text_encoder_layer (50→49), min_duration,
    _execution_device, audio_scheduler
  - VRAM-saving transformer patches from inference.py are applied here
    so the pipeline generates clean video (not blocky artifacts)
  - num_frames is snapped to 17*n+5 in __call__
"""

import gc
import logging
import types
import torch
from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")


class WeeMiniMaxPipeline(WeeVideoPipeline):

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from pathlib import Path

        # ── Stub injection ────────────────────────────────────────────────────
        import transformers as _tf
        import diffusers as _df
        for _cls in ("MiniMaxH3Qwen3VLHFEncoder",):
            if not hasattr(_tf, _cls):
                setattr(_tf, _cls, type(_cls, (), {}))
        for _cls in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE", "MiniMaxH3DiTModel"):
            if not hasattr(_df, _cls):
                setattr(_df, _cls, type(_cls, (), {}))

        # ── Resolve model dir ─────────────────────────────────────────────────
        from weellm.io.utils import resolve_model_path
        model_dir_str  = str(resolve_model_path(str(model_dir)))
        model_dir_path = Path(model_dir_str)

        device = kwargs.get("device",      "cuda")
        dtype  = kwargs.get("torch_dtype", torch.bfloat16)

        # ── Audio VAE is loaded naturally ─────────────────────────────────────

        # ── Standard WeeBasePipeline loading ─────────────────────────────────
        # This loads VAE → TE (via GGUF override_weights_path) → Transformer
        pipe = super().from_pretrained(model_dir_str, **kwargs)
        underlying = pipe._pipeline
        
        # Diffusers 0.41 ModularPipeline expects 'vae' in components, not 'video_vae'
        if getattr(underlying, "vae", None) is None and hasattr(underlying, "video_vae"):
            underlying.vae = underlying.video_vae
            if hasattr(underlying, "components"):
                underlying.components["vae"] = underlying.video_vae
        
        # ── Apply post-load patches ───────────────────────────────────────────
        cls._apply_pipeline_patches(underlying, device)
        cls._inject_tokenizer_processor(underlying, model_dir_path)
        cls._inject_audio_scheduler(underlying)
        cls._apply_vram_patches(underlying)

        return pipe

    def __call__(self, prompt: str, **kwargs):
        # Strip unsupported kwargs
        for _k in ("guidance_scale", "negative_prompt", "image_guidance_scale"):
            kwargs.pop(_k, None)

        # Snap num_frames to 17*n+5 (MiniMax VAE requirement)
        _nf = kwargs.get("num_frames")
        if _nf is not None:
            _nf = int(_nf)
            while _nf % 17 != 5:
                _nf += 1
            if _nf != kwargs["num_frames"]:
                logger.info("[WeeLLM/MiniMax] num_frames snapped %d → %d (must be 17n+5).",
                            kwargs["num_frames"], _nf)
            kwargs["num_frames"] = _nf

        return super().__call__(prompt=prompt, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Patch helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_pipeline_patches(underlying, device: str):
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

        # text_encoder_layer: 50 → 49  (pruned 50-layer model, indices 0-49)
        if not getattr(MiniMaxH3ModularPipeline, "_weellm_te_layer_patched", False):
            MiniMaxH3ModularPipeline.text_encoder_layer = property(lambda self: 49)
            MiniMaxH3ModularPipeline._weellm_te_layer_patched = True
            logger.info("[WeeLLM/MiniMax] text_encoder_layer: 50 → 49")

        # min_duration: 5.0s → 0.0s  (allow any frame count)
        if not getattr(MiniMaxH3ModularPipeline, "_weellm_min_dur_patched", False):
            MiniMaxH3ModularPipeline.min_duration = property(lambda self: 0.0)
            MiniMaxH3ModularPipeline._weellm_min_dur_patched = True
            logger.info("[WeeLLM/MiniMax] min_duration: 5.0s → 0.0s")

        # _execution_device → real GPU
        real_device = torch.device(device)
        if not getattr(MiniMaxH3ModularPipeline, "_weellm_exec_dev_patched", False):
            MiniMaxH3ModularPipeline._execution_device = property(lambda self: real_device)
            MiniMaxH3ModularPipeline._weellm_exec_dev_patched = True
            logger.info("[WeeLLM/MiniMax] _execution_device → %s", real_device)

        # Video scheduler shift = 12.0
        if getattr(underlying, "scheduler", None) is not None:
            try:
                from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
                if isinstance(underlying.scheduler, MiniMaxH3Scheduler):
                    if getattr(underlying.scheduler, "shift", None) != 12.0:
                        underlying.scheduler = MiniMaxH3Scheduler(shift=12.0)
                        logger.info("[WeeLLM/MiniMax] scheduler shift reset to 12.0")
            except Exception:
                pass

    @staticmethod
    def _inject_tokenizer_processor(underlying, model_dir_path):
        try:
            te_dir = model_dir_path / "text_encoder"
            from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
            if getattr(underlying, "tokenizer", None) is None:
                underlying.tokenizer = Qwen2TokenizerFast.from_pretrained(str(te_dir))
            if getattr(underlying, "processor", None) is None:
                underlying.processor = Qwen3VLProcessor.from_pretrained(str(te_dir))
            logger.info("[WeeLLM/MiniMax] tokenizer + processor injected from text_encoder/.")
        except Exception as e:
            logger.warning("[WeeLLM/MiniMax] Could not load tokenizer/processor: %s", e)

    @staticmethod
    def _inject_audio_scheduler(underlying):
        if getattr(underlying, "audio_scheduler", None) is None:
            try:
                from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
                underlying.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
                logger.info("[WeeLLM/MiniMax] audio_scheduler injected (shift=3.0).")
            except Exception as e:
                logger.warning("[WeeLLM/MiniMax] audio_scheduler inject failed: %s", e)

    @staticmethod
    def _apply_vram_patches(underlying):
        """
        Apply the same VRAM-reduction patches as inference.py:
          Strategy 1: Chunked FFN + in-place AdaLN in transformer blocks
          Strategy 2: Early del norm_hidden_states + in-place QK-norm + RoPE
          Strategy 3: In-place index_copy_ + explicit embed deletion in transformer forward
        These are required to keep peak VRAM within budget on 4 GB GPUs.
        """
        transformer = getattr(underlying, "transformer", None)
        if transformer is None:
            logger.warning("[WeeLLM/MiniMax] No transformer found — skipping VRAM patches.")
            return

        try:
            from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
            FFN_CHUNK_SIZE = 1024

            def _chunked_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

                # Attention with in-place AdaLN
                residual = hidden_states
                norm_hidden_states = self.norm1(hidden_states)
                scale_indexed = scale_msa.index_select(0, adaln_indices); scale_indexed.add_(1.0)
                norm_hidden_states.mul_(scale_indexed); del scale_indexed
                shift_indexed = shift_msa.index_select(0, adaln_indices)
                norm_hidden_states.add_(shift_indexed); del shift_indexed
                attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
                gate_msa_col = gate_msa.index_select(0, adaln_indices)
                attn_output.mul_(gate_msa_col); del gate_msa_col
                hidden_states = residual.add_(attn_output); del attn_output

                # FFN with in-place AdaLN + chunked computation
                residual = hidden_states
                norm_hidden_states = self.norm2(hidden_states)
                scale_indexed = scale_mlp.index_select(0, adaln_indices); scale_indexed.add_(1.0)
                norm_hidden_states.mul_(scale_indexed); del scale_indexed
                shift_indexed = shift_mlp.index_select(0, adaln_indices)
                norm_hidden_states.add_(shift_indexed); del shift_indexed
                seq_len = norm_hidden_states.shape[1]
                ff_output = torch.empty_like(norm_hidden_states)
                for i in range(0, seq_len, FFN_CHUNK_SIZE):
                    end = min(i + FFN_CHUNK_SIZE, seq_len)
                    ff_output[:, i:end, :] = self.ff(norm_hidden_states[:, i:end, :])
                del norm_hidden_states
                gate_col = gate_mlp.index_select(0, adaln_indices)
                ff_output.mul_(gate_col); del gate_col
                hidden_states = residual.add_(ff_output); del ff_output
                return hidden_states

            patched = 0
            for module in transformer.modules():
                if isinstance(module, MiniMaxH3TransformerBlock):
                    module.forward = types.MethodType(_chunked_block_forward, module)
                    patched += 1
            logger.info("[WeeLLM/MiniMax] Strategy 1: Chunked FFN + in-place AdaLN patched into %d blocks.", patched)
        except Exception as e:
            logger.warning("[WeeLLM/MiniMax] Strategy 1 patch failed: %s", e)

        try:
            from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
            from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor

            def _apply_rotary_emb_inplace_(hs, cos, sin):
                rotary_dim = cos.shape[-1]; half = rotary_dim // 2
                cos_ = cos.to(hs.dtype)[None, :, None, :]
                sin_ = sin.to(hs.dtype)[None, :, None, :]
                hs_r = hs[..., :rotary_dim]
                x1 = hs_r[..., :half].clone(); x2 = hs_r[..., half:].clone()
                temp = x2 * sin_[..., :half]
                hs_r[..., :half].copy_(x1).mul_(cos_[..., :half]).sub_(temp); del temp
                temp = x1 * sin_[..., half:]
                hs_r[..., half:].copy_(x2).mul_(cos_[..., half:]).add_(temp); del temp
                del x1, x2
                return hs

            def _rmsnorm_inplace_(x, weight, eps):
                rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt().to(x.dtype)
                x.div_(rms); x.mul_(weight); return x

            def _mem_efficient_attn_call(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
                if attn.fused_projections:
                    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                else:
                    query = attn.to_q(hidden_states)
                    key   = attn.to_k(hidden_states)
                    value = attn.to_v(hidden_states)
                del hidden_states
                query = query.unflatten(-1, (attn.heads, -1))
                key   = key.unflatten(-1, (attn.heads, -1))
                value = value.unflatten(-1, (attn.heads, -1))
                _rmsnorm_inplace_(query, attn.norm_q.weight, attn.norm_q.eps)
                _rmsnorm_inplace_(key,   attn.norm_k.weight, attn.norm_k.eps)
                if rotary_emb is not None:
                    _apply_rotary_emb_inplace_(query, *rotary_emb)
                    _apply_rotary_emb_inplace_(key,   *rotary_emb)
                hidden_states = dispatch_attention_fn(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                    is_causal=False, backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                query_dtype = query.dtype
                del query, key, value
                hidden_states = hidden_states.flatten(2, 3).to(query_dtype)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

            MiniMaxH3AttnProcessor.__call__ = _mem_efficient_attn_call
            MiniMaxH3AttnProcessor._attention_backend = AttentionBackendName._NATIVE_EFFICIENT
            logger.info("[WeeLLM/MiniMax] Strategy 2: In-place QK-norm + RoPE + early del norm_hs patched.")
        except Exception as e:
            logger.warning("[WeeLLM/MiniMax] Strategy 2 patch failed: %s", e)

        try:
            from diffusers.models.transformers.transformer_minimax_h3 import (
                MiniMaxH3Transformer3DModel, MiniMaxH3TransformerOutput, MINIMAX_H3_MODALITY_NUM,
            )
            from diffusers.models.modeling_utils import get_parameter_dtype as _get_param_dtype

            def _mem_efficient_transformer_forward(
                self, hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids,
                video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True,
            ):
                if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
                    raise ValueError(f"`position_ids` must be `(seq_len, 3)`, got {list(position_ids.shape)}.")
                sequence_length = position_ids.shape[0]
                if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
                    raise ValueError("`token_tags` and `timestep_indices` must be `(seq_len,)` tensors.")

                rotary_emb = self.rope(position_ids)
                target_device = hidden_states.device

                # JIT-move setup modules
                for m in (self.proj_in, self.audio_proj_in, self.context_embedder, self.token_refiner):
                    m.to(target_device)

                video_embeds = self.proj_in(hidden_states.to(_get_param_dtype(self.proj_in)))
                audio_embeds = self.audio_proj_in(audio_hidden_states.to(_get_param_dtype(self.audio_proj_in)))
                text_embeds  = self.context_embedder(encoder_hidden_states.to(_get_param_dtype(self.context_embedder)))
                text_embeds  = self.token_refiner(text_embeds)

                # Offload setup modules back
                for m in (self.proj_in, self.audio_proj_in, self.context_embedder, self.token_refiner):
                    m.to("cpu")

                # In-place packing (avoids 231 MB duplicate allocations)
                hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
                hidden_states.index_copy_(1, text_indices,  text_embeds)
                hidden_states.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
                hidden_states.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))

                # Free embeddings now before the block loop
                del video_embeds, audio_embeds, text_embeds
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                temb         = self.time_proj(timestep)
                temb         = self.time_embedder(temb.to(_get_param_dtype(self.time_embedder)))
                adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

                for block in self.transformer_blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        hidden_states = self._gradient_checkpointing_func(
                            block, hidden_states, temb, adaln_indices, rotary_emb)
                    else:
                        hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)

                hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(_get_param_dtype(self.proj_out))
                video_output  = self.proj_out(hidden_states).index_select(1, video_indices)
                audio_output  = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

                if not return_dict:
                    return (video_output, audio_output)
                return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

            MiniMaxH3Transformer3DModel.forward = _mem_efficient_transformer_forward
            logger.info("[WeeLLM/MiniMax] Strategy 3: In-place packing + embed deletion + peak reset patched.")
        except Exception as e:
            logger.warning("[WeeLLM/MiniMax] Strategy 3 patch failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Stub
# ─────────────────────────────────────────────────────────────────────────────

class _DummyAudioVAE:
    """No-op stub for the audio VAE (video-only mode)."""
    class config:
        latent_channels = 32
        sampling_rate   = 32000
        latents_mean    = [0.0] * 32
        latents_std     = [1.0] * 32

    def __call__(self, *a, **kw): return None
    def to(self, *a, **kw):       return self

    def encode(self, *a, **kw):
        class _D:
            def mode(self):    return torch.zeros(1, 32, 1)
            def sample(self, generator=None): return torch.zeros(1, 32, 1)
        class _O: latent_dist = _D()
        return _O()

    def decode(self, *a, **kw):
        return (torch.zeros(2, 1, 1),)
