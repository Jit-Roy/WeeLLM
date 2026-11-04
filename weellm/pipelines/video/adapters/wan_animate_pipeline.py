"""
weellm.pipelines.video.adapters.wan_animate_pipeline
====================================================
Adapter for Wan-Animate video generation models (e.g. Wan2.2-Animate-2-14B-Diffusers).
Supports character animation from reference image and driving motion video with
layer-streaming memory optimization for 4GB GPUs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, List

import PIL.Image
import torch

from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline, WeeVideoResult
from weellm.pipelines.video.adapters.wan_pipeline import DEFAULT_WAN_NEG_PROMPT

logger = logging.getLogger("weellm")


def _load_image(img):
    if img is None:
        return None
    if isinstance(img, str):
        from PIL import Image, ImageOps
        return ImageOps.exif_transpose(Image.open(img)).convert("RGB")
    if isinstance(img, list) and len(img) > 0 and isinstance(img[0], str):
        from PIL import Image, ImageOps
        return [ImageOps.exif_transpose(Image.open(x)).convert("RGB") for x in img]
    return img


def _load_video_frames(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            from diffusers.utils import load_video
            return load_video(v)
        except Exception:
            from PIL import Image
            import cv2
            cap = cv2.VideoCapture(v)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            cap.release()
            return frames
    return v


def _patch_flex_attention():
    """Ensure flex_attention works safely within memory budget without Triton (e.g. Windows)."""
    try:
        import torch.nn.attention.flex_attention as fa
        if not hasattr(fa, "_weellm_uncompiled_patched"):
            orig_create_block_mask = fa.create_block_mask

            def _safe_create_block_mask(*args, **kwargs):
                try:
                    import triton  # noqa: F401
                    has_triton = True
                except Exception:
                    has_triton = False

                if not has_triton:
                    kwargs["_compile"] = False
                try:
                    return orig_create_block_mask(*args, **kwargs)
                except Exception:
                    if kwargs.get("_compile", True):
                        kwargs["_compile"] = False
                        return orig_create_block_mask(*args, **kwargs)
                    raise

            fa.create_block_mask = _safe_create_block_mask
            fa._weellm_uncompiled_patched = True

            try:
                import diffusers.models.transformers.transformer_wan_animate_2 as tw
                tw.create_block_mask = _safe_create_block_mask
            except Exception:
                pass

        if not hasattr(fa, "_weellm_chunked_flex_patched"):
            orig_flex = fa.flex_attention
            try:
                compiled_flex = torch.compile(orig_flex)
            except Exception:
                compiled_flex = None

            def _safe_flex_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
                if compiled_flex is not None:
                    try:
                        return compiled_flex(query, key, value, **kwargs)
                    except Exception as e:
                        logger.debug("[WeeLLM] compiled flex_attention failed (%s), using chunked fallback.", e)

                HEAD_CHUNK = 1
                num_heads = query.shape[1]
                if num_heads > HEAD_CHUNK:
                    out = torch.empty_like(query)
                    for i in range(num_heads):
                        out[:, i : i + 1] = orig_flex(
                            query[:, i : i + 1],
                            key[:, i : i + 1],
                            value[:, i : i + 1],
                            **kwargs
                        )
                    return out
                return orig_flex(query, key, value, **kwargs)

            fa.flex_attention = _safe_flex_attention
            fa._weellm_chunked_flex_patched = True
            logger.info("[WeeLLM] Enabled Triton compiled flex_attention with head-chunked fallback.")

        try:
            import diffusers.models.transformers.transformer_wan_animate_2 as tw
            if not hasattr(tw, "_weellm_rope_patched"):
                def _safe_rope_apply(x, grid_sizes, freqs, time_stride=1):
                    n, c = x.size(2), x.size(3) // 2
                    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
                    # Pre-allocate output in the original dtype (bfloat16) to avoid
                    # accumulating large float32 intermediate tensors. Each chunk is
                    # computed in float32 (required by view_as_complex) and immediately
                    # cast+written into `result`, so peak VRAM = 1 chunk (~41 MB) +
                    # result (~158 MB bf16) ≈ 199 MB — fits on 4 GB GPU.
                    # ROPE_CHUNK = 2048 tokens: 2048 × n_heads × head_dim/2 × 8 B ≈ 41 MB
                    ROPE_CHUNK = 2048
                    result = torch.empty_like(x)
                    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                        seq_len = f * h * w
                        # freqs_i: [seq_len, 1, c/2] complex64 — small (~7 MB), built once
                        freqs_i = torch.cat(
                            [
                                freqs_split[0][: f * time_stride : time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
                                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                            ],
                            dim=-1,
                        ).reshape(seq_len, 1, -1).to(device=x.device, dtype=torch.complex64)

                        for s in range(0, seq_len, ROPE_CHUNK):
                            e = min(s + ROPE_CHUNK, seq_len)
                            # view_as_complex requires float32 or float64
                            x_chunk = torch.view_as_complex(
                                x[i, s:e].float().reshape(e - s, n, -1, 2)
                            )
                            rot = x_chunk * freqs_i[s:e]
                            del x_chunk
                            # Write directly into result in target dtype — no accumulation
                            result[i, s:e] = torch.view_as_real(rot).flatten(2).to(x.dtype)
                            del rot

                        del freqs_i
                        # Padding tokens are copied unchanged
                        if seq_len < result.shape[1]:
                            result[i, seq_len:] = x[i, seq_len:]

                    return result

                tw.rope_apply = _safe_rope_apply
                tw._weellm_rope_patched = True

            if not hasattr(tw, "_weellm_kv_cache_patched"):
                class CPUPinnedWanAnimate2KVLayerCache:
                    """CPU-backed KV cache to prevent storing gigabytes of KV across 40 layers in VRAM."""
                    def __init__(self):
                        self.key: torch.Tensor | None = None
                        self.value: torch.Tensor | None = None

                    def store(self, key: torch.Tensor, value: torch.Tensor):
                        self.key = key.to("cpu")
                        self.value = value.to("cpu")

                    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
                        if self.key is None:
                            raise RuntimeError("The KV cache is empty. Run the reference pass before the generation pass.")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        return self.key.to(device, non_blocking=True), self.value.to(device, non_blocking=True)

                    def clear(self):
                        self.key = None
                        self.value = None

                tw.WanAnimate2KVLayerCache = CPUPinnedWanAnimate2KVLayerCache
                tw._weellm_kv_cache_patched = True
                logger.info("[WeeLLM] Enabled CPU-backed KV cache for Wan Animate layers.")

            if not hasattr(tw, "_weellm_attn_proc_patched"):
                # ── Head-chunked cached attention ──────────────────────────────────────────
                # At 960×544: full packed key = (1, 87808, 40, 128) × 2 bytes = 899 MB each
                # — impossible on 4 GB VRAM.  Processing _HC heads at a time keeps the peak
                # packed allocation at (_HC/40) × 899 MB ≈ 90 MB (for _HC=4).
                # Reference KV is moved from CPU head-group by head-group; the full 439 MB
                # GPU copy is never needed simultaneously.
                _HC = 4   # heads per chunk (4 × 128 dim × ~88K tokens × 2 byte = 90 MB each)

                import math as _math

                _orig_attn_proc_call = tw.WanAnimate2AttnProcessor.__call__

                def _hc_attn_proc_call(
                    _self, attn, hidden_states, rotary_emb, grid_sizes, kv_cache,
                    kv_cache_mode, rope_stride=1, reference_rotary_emb=None,
                    reference_grid_sizes=None, reference_rope_stride=1,
                    attention_mask=None, origin_latent_frames=None, origin_latent_hw=None,
                ):
                    if kv_cache_mode != "cached":
                        # Extract mode: original code (no large packed buffers needed)
                        return _orig_attn_proc_call(
                            _self, attn, hidden_states, rotary_emb, grid_sizes, kv_cache,
                            kv_cache_mode, rope_stride, reference_rotary_emb,
                            reference_grid_sizes, reference_rope_stride,
                            attention_mask, origin_latent_frames, origin_latent_hw,
                        )

                    # ── Cached mode: head-chunked path ────────────────────────────────────
                    from diffusers.models.attention_dispatch import (
                        dispatch_attention_fn,
                        AttentionBackendName,
                    )

                    # QKV projections (replicate _get_qkv_projections)
                    if getattr(attn, "fused_projections", False):
                        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                    else:
                        query = attn.to_q(hidden_states)
                        key   = attn.to_k(hidden_states)
                        value = attn.to_v(hidden_states)
                    query = attn.norm_q(query)
                    key   = attn.norm_k(key)
                    query = query.unflatten(2, (attn.heads, -1))
                    key   = key.unflatten(2, (attn.heads, -1))
                    value = value.unflatten(2, (attn.heads, -1))

                    device = query.device
                    batch_size, _seq_len, heads, head_dim = query.shape

                    frames, height, width = grid_sizes[0].tolist()
                    ref_frames, ref_height, ref_width = reference_grid_sizes[0].tolist()
                    hw         = height * width
                    ref_hw     = ref_height * ref_width
                    valid_len  = frames * hw
                    ref_valid  = ref_frames * ref_hw

                    packed_len     = _math.ceil((origin_latent_frames + 1) * origin_latent_hw / 128) * 128
                    packed_ref_len = _math.ceil(origin_latent_frames * origin_latent_hw / 128) * 128
                    gen_slice = slice(0, frames * origin_latent_hw)
                    ref_slice = slice(packed_len, packed_len + ref_frames * origin_latent_hw)

                    # Apply RoPE to generation Q and K for all heads (cheap — float32 chunk already
                    # handled by _safe_rope_apply).  Key_ref gets rope per head-group below.
                    query_r = tw.rope_apply(query, grid_sizes, rotary_emb, rope_stride).type_as(value)
                    key_r   = tw.rope_apply(key,   grid_sizes, rotary_emb, rope_stride).type_as(value)

                    # Reference KV lives on CPU (stored there by CPUPinnedWanAnimate2KVLayerCache).
                    # Access the raw CPU tensors directly to avoid a full 439 MB H2D transfer.
                    key_ref_cpu = kv_cache.key    # (batch, ref_tokens, heads, head_dim) on CPU
                    val_ref_cpu = kv_cache.value

                    # Output accumulator — valid tokens only, filled head-group by head-group
                    hidden_out = query.new_zeros(batch_size, valid_len, heads, head_dim)

                    for h0 in range(0, heads, _HC):
                        h1 = min(h0 + _HC, heads)
                        g  = h1 - h0

                        # Small packed buffers for g heads (~90 MB each for 960×544)
                        q_h = query_r.new_zeros(batch_size, packed_len, g, head_dim)
                        k_h = key_r.new_zeros(batch_size, packed_len + packed_ref_len, g, head_dim)
                        v_h = value.new_zeros(batch_size, packed_len + packed_ref_len, g, head_dim)

                        # Scatter generation tokens into packed layout
                        q_h[:, gen_slice].view(batch_size, frames, origin_latent_hw, g, head_dim)[:, :, :hw] = (
                            query_r[:, :valid_len, h0:h1, :].view(batch_size, frames, hw, g, head_dim)
                        )
                        k_h[:, gen_slice].view(batch_size, frames, origin_latent_hw, g, head_dim)[:, :, :hw] = (
                            key_r[:, :valid_len, h0:h1, :].view(batch_size, frames, hw, g, head_dim)
                        )
                        v_h[:, gen_slice].view(batch_size, frames, origin_latent_hw, g, head_dim)[:, :, :hw] = (
                            value[:, :valid_len, h0:h1, :].view(batch_size, frames, hw, g, head_dim)
                        )

                        # Pull just g heads of ref KV from CPU → GPU and apply RoPE
                        k_ref_g = key_ref_cpu[:, :, h0:h1, :].to(device, non_blocking=True)
                        v_ref_g = val_ref_cpu[:, :, h0:h1, :].to(device, non_blocking=True)
                        k_ref_g = tw.rope_apply(
                            k_ref_g, reference_grid_sizes,
                            reference_rotary_emb, reference_rope_stride,
                        ).type_as(value)

                        # Scatter reference tokens into packed layout
                        k_h[:, ref_slice].view(batch_size, ref_frames, origin_latent_hw, g, head_dim)[:, :, :ref_hw] = (
                            k_ref_g[:, :ref_valid].view(batch_size, ref_frames, ref_hw, g, head_dim)
                        )
                        v_h[:, ref_slice].view(batch_size, ref_frames, origin_latent_hw, g, head_dim)[:, :, :ref_hw] = (
                            v_ref_g[:, :ref_valid].view(batch_size, ref_frames, ref_hw, g, head_dim)
                        )
                        del k_ref_g, v_ref_g

                        # Flex attention for this head group (same block mask as full version)
                        out_h = dispatch_attention_fn(
                            q_h, k_h, v_h,
                            attn_mask=attention_mask,
                            backend=AttentionBackendName.FLEX,
                            parallel_config=_self._parallel_config,
                        )
                        del q_h, k_h, v_h

                        # Gather valid output tokens
                        hidden_out[:, :, h0:h1, :] = (
                            out_h[:, gen_slice]
                            .view(batch_size, frames, origin_latent_hw, g, head_dim)[:, :, :hw]
                            .reshape(batch_size, valid_len, g, head_dim)
                        )
                        del out_h

                    del query_r, key_r

                    # Append unattended padding positions unchanged (original behaviour)
                    hidden_states = torch.cat([hidden_out, query[:, valid_len:]], dim=1)
                    del hidden_out

                    # to_out projection (same as original lines 276-278)
                    hidden_states = hidden_states.flatten(2, 3).type_as(query)
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                    return hidden_states

                tw.WanAnimate2AttnProcessor.__call__ = _hc_attn_proc_call
                tw._weellm_attn_proc_patched = True
                logger.info(
                    "[WeeLLM] Enabled head-chunked cached attention (HC=%d heads/iter) "
                    "for Wan Animate — peak packed KV ≈ %d×%d×%d×2 bytes per chunk.",
                    _HC, 88000, _HC, 128,
                )

            if not hasattr(tw, "_weellm_block_fwd_patched"):

                def _mem_efficient_block_forward(
                    self,
                    hidden_states: torch.Tensor,
                    temb: torch.Tensor,
                    encoder_hidden_states: torch.Tensor,
                    kv_cache,
                    kv_cache_mode: str,
                    rotary_emb: torch.Tensor,
                    grid_sizes: torch.Tensor,
                    encoder_hidden_states_image=None,
                    reference_rotary_emb=None,
                    reference_grid_sizes=None,
                    attention_mask=None,
                    origin_latent_frames=None,
                    origin_latent_hw=None,
                ):
                    """
                    Memory-efficient replacement for WanAnimate2TransformerBlock.forward.
                    The stock code does (hidden_states.float() + small) which creates a
                    full float32 copy of hidden_states (330 MB on 4 GB GPU → OOM).
                    We instead do hidden_states + small.type_as(hidden_states), which
                    only upcasts the small addend and avoids the large allocation.
                    """
                    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                        self.modulation + temb
                    ).chunk(6, dim=1)

                    # 1. Self-attention
                    # Stock: (hidden_states.float() * scale + shift).type_as — norm output is small, fine
                    norm_hs = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
                    attn_output = self.self_attn(
                        norm_hs,
                        rotary_emb=rotary_emb,
                        grid_sizes=grid_sizes,
                        kv_cache=kv_cache,
                        kv_cache_mode=kv_cache_mode,
                        rope_stride=self.refer_stride if kv_cache_mode == "extract" else 1,
                        reference_rotary_emb=reference_rotary_emb,
                        reference_grid_sizes=reference_grid_sizes,
                        reference_rope_stride=self.refer_stride,
                        attention_mask=attention_mask,
                        origin_latent_frames=origin_latent_frames,
                        origin_latent_hw=origin_latent_hw,
                    )
                    del norm_hs
                    # Stock: (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
                    # Fix: avoid float32 copy of full hidden_states (330 MB), cast only the small addend
                    hidden_states = hidden_states + (attn_output * gate_msa).type_as(hidden_states)
                    del attn_output

                    # 2. Cross-attention
                    # norm3(hidden_states.float()) is fine — norm output is small
                    norm_hs = self.norm3(hidden_states.float()).type_as(hidden_states)
                    cross_out = self.cross_attn(
                        norm_hs,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_image=encoder_hidden_states_image,
                    )
                    del norm_hs
                    hidden_states = hidden_states + cross_out
                    del cross_out

                    # 3. Feed-forward
                    norm_hs = (self.norm2(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
                        hidden_states
                    )
                    ff_output = self.ffn(norm_hs)
                    del norm_hs
                    # Stock: (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
                    # Fix: cast only ff_output * c_gate_msa (small), not all of hidden_states
                    hidden_states = hidden_states + (ff_output.float() * c_gate_msa).type_as(hidden_states)
                    del ff_output

                    return hidden_states

                tw.WanAnimate2TransformerBlock.forward = _mem_efficient_block_forward
                tw._weellm_block_fwd_patched = True
                logger.info("[WeeLLM] Enabled memory-efficient WanAnimate2TransformerBlock forward (no large float32 upcasts).")

            try:
                import diffusers.modular_pipelines.wan_animate_2.video_processor as wvp
                if not hasattr(wvp, "_weellm_fill_patched"):
                    def _adaptive_resize_and_fill(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
                        import numpy as np
                        from diffusers.utils import PIL_INTERPOLATION

                        ratio = width / height
                        src_ratio = image.width / image.height

                        src_w = width if ratio < src_ratio else image.width * height // image.height
                        src_h = height if ratio >= src_ratio else image.height * width // image.width

                        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])

                        fill = getattr(self.config, "fill_color", 0)
                        if fill == 0 or fill is None:
                            arr = np.array(image.convert("RGB"))
                            border_pixels = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
                            pad_color = tuple(np.median(border_pixels, axis=0).astype(int).tolist())
                        else:
                            pad_color = fill

                        res = PIL.Image.new("RGB", (width, height), color=pad_color)
                        res.paste(resized, box=((width - src_w) // 2, (height - src_h) // 2))
                        return res

                    wvp.WanAnimate2VideoProcessor._resize_and_fill = _adaptive_resize_and_fill
                    wvp._weellm_fill_patched = True
                    logger.info("[WeeLLM] Enabled edge-adaptive letterbox padding for Wan Animate.")
            except Exception as e:
                logger.debug("[WeeLLM] video processor fill patch skipped: %s", e)
        except Exception as e:
            logger.debug("[WeeLLM] rope_apply or kv_cache patch skipped: %s", e)
    except Exception as e:
        logger.debug("[WeeLLM] flex_attention safe patch skipped: %s", e)


class WeeWanAnimatePipeline(WeeVideoPipeline):
    """
    WeeLLM Pipeline adapter for Wan Animate video generation models
    (e.g., Wan2.2-Animate-2-14B-Diffusers).
    Supports character animation with layer-streaming memory optimization
    and complete WeeLLM caching (text encoder, VAE, denoise steps, and final video).
    """

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **kwargs) -> "WeeWanAnimatePipeline":
        _patch_flex_attention()
        skip_components = {
            k[:-5] for k, v in kwargs.items()
            if k.endswith("_path") and v is not None
        }
        skip_components.update({k for k, v in kwargs.items() if v is None})

        from weellm.io.utils import resolve_model_path
        model_dir_str = str(resolve_model_path(str(model_dir), skip_components=skip_components or None))
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("torch_dtype", torch.bfloat16)

        pipe = super().from_pretrained(model_dir_str, **kwargs)
        underlying = pipe._pipeline

        # Ensure model_dir is set on pipeline instances
        setattr(pipe, "model_dir", model_dir_str)
        setattr(underlying, "model_dir", model_dir_str)

        # Ensure VAE is in target dtype (bfloat16 if GPU natively supports it)
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        vae = getattr(underlying, "vae", None)
        if vae is not None and getattr(vae, "dtype", None) != target_dtype:
            vae.to(dtype=target_dtype)
        vae_streamer = getattr(pipe, "vae_streamer", None)
        if vae_streamer is not None:
            vae_streamer.dtype = target_dtype

        # Modular pipeline expects image_processor, video_processor and guider
        try:
            from diffusers.modular_pipelines.wan_animate_2.video_processor import WanAnimate2VideoProcessor
            # WanAnimate2ProcessImagesInputStep requires get_default_height_width from image_processor
            if not hasattr(getattr(underlying, "image_processor", None), "get_default_height_width"):
                video_proc = WanAnimate2VideoProcessor(resample="bicubic")
                underlying.image_processor = video_proc
                if hasattr(underlying, "components"):
                    underlying.components["image_processor"] = video_proc

            if getattr(underlying, "video_processor", None) is None:
                underlying.video_processor = WanAnimate2VideoProcessor()
                if hasattr(underlying, "components"):
                    underlying.components["video_processor"] = underlying.video_processor
        except Exception as e:
            logger.debug("[WeeLLM/WanAnimate] processor auto-init: %s", e)

        if getattr(underlying, "guider", None) is None:
            try:
                from diffusers.guiders import ClassifierFreeGuidance
                underlying.guider = ClassifierFreeGuidance(guidance_scale=3.0)
                if hasattr(underlying, "components"):
                    underlying.components["guider"] = underlying.guider
            except Exception as e:
                logger.debug("[WeeLLM/WanAnimate] guider auto-init: %s", e)

        # Ensure execution device is accurately reported on modular pipeline
        real_device = torch.device(device)
        underlying_cls = underlying.__class__
        if not hasattr(underlying_cls, "_weellm_exec_dev_patched"):
            underlying_cls._execution_device = property(lambda self_obj: real_device)
            underlying_cls.device = property(lambda self_obj: real_device)
            underlying_cls._weellm_exec_dev_patched = True

        return pipe

    def __call__(self, prompt: str = "", **kwargs):
        raw_first_frame = kwargs.get("first_frame", kwargs.get("image", None))
        raw_driving_video = kwargs.get("driving_video", kwargs.get("video", kwargs.get("motion_video", None)))

        # 1. Conditioning image resolution (accept either 'first_frame' or 'image')
        first_frame = kwargs.pop("first_frame", None)
        if first_frame is None:
            first_frame = kwargs.pop("image", None)
        else:
            kwargs.pop("image", None)

        first_frame = _load_image(first_frame)

        # 2. Driving motion video resolution
        driving_video = kwargs.pop("driving_video", kwargs.pop("video", kwargs.pop("motion_video", None)))
        driving_video = _load_video_frames(driving_video)
        driving_video_fps = kwargs.pop("driving_video_fps", None)

        num_frames = kwargs.pop("num_frames", None)
        if num_frames is None and "segment_frame_length" in kwargs:
            num_frames = kwargs.get("segment_frame_length")

        if driving_video is not None and num_frames is not None:
            if len(driving_video) > int(num_frames):
                logger.info(
                    "[WeeLLM/WanAnimate] Slicing driving video from %d frames to num_frames=%d frames.",
                    len(driving_video), int(num_frames),
                )
                driving_video = driving_video[: int(num_frames)]

        if driving_video is None:
            if first_frame is not None:
                ref_img = first_frame[0] if isinstance(first_frame, list) else first_frame
                driving_video = [ref_img] * int(num_frames or 81)
                logger.debug("[WeeLLM/WanAnimate] No driving video provided; using static first frame as driving motion.")
            else:
                raise ValueError(
                    "[WeeLLM/WanAnimate] Wan Animate requires a character image ('first_frame' or 'image') "
                    "and optionally a driving motion video ('driving_video' or 'video')."
                )

        # 3. Dimensions snapping (Wan spatial patches are 16-multiple)
        target_h = kwargs.pop("height", 800)
        target_w = kwargs.pop("width", 640)
        target_h = max(32, round(target_h / 16) * 16)
        target_w = max(32, round(target_w / 16) * 16)

        # 4. Settings & cache controls
        neg_prompt = kwargs.pop("negative_prompt", None) or DEFAULT_WAN_NEG_PROMPT
        num_inference_steps = kwargs.pop("num_inference_steps", 20)
        guidance_scale = kwargs.pop("guidance_scale", 3.0)
        fps = kwargs.pop("fps", 24)
        segment_frame_length = kwargs.pop("segment_frame_length", 81)
        prev_segment_conditioning_frames = kwargs.pop("prev_segment_conditioning_frames", 1)
        generator = kwargs.pop("generator", None)
        seed = kwargs.pop("seed", None)
        if generator is None and seed is not None:
            device_str = str(getattr(self, "device", "cuda"))
            generator = torch.Generator(device=device_str).manual_seed(int(seed))

        no_cache = kwargs.pop("no_cache", False)
        fresh = kwargs.pop("fresh", False)
        cache_every = kwargs.pop("cache_every", 1)
        lora_weights = kwargs.pop("lora_weights", None)

        # Clean up any leftover kwargs
        for k in ["last_frame", "last_image", "video", "first_frame", "image"]:
            kwargs.pop(k, None)

        underlying = getattr(self, "_pipeline", self)
        guider = getattr(underlying, "guider", None)
        if guider is not None and hasattr(guider, "guidance_scale"):
            guider.guidance_scale = guidance_scale

        from weellm.io.memory import evict_module
        from weellm.io.utils import clean_memory

        # ── Setup VideoStepCache ──────────────────────────────────────────────
        video_cache = None
        if not no_cache:
            video_cache = self._setup_cache(
                prompt=prompt,
                height=target_h,
                width=target_w,
                num_frames=int(num_frames or 81),
                steps=num_inference_steps,
                seed=seed if seed is not None else 42,
                negative_prompt=neg_prompt,
                save_every=cache_every,
                lora_weights=lora_weights,
                first_frame=raw_first_frame or first_frame,
                last_frame=raw_driving_video or driving_video,
            )
            if fresh:
                video_cache.clear_run()

        # ── Fast-path 1: Final Video Output Cache Hit ─────────────────────────
        if video_cache is not None and video_cache.has_final():
            logger.info("[VideoCache] Final video cache HIT (%s) — restoring generated video from disk!", video_cache.run_dir.name)
            try:
                cached_final = video_cache.load_final()
                if "videos" in cached_final:
                    return WeeVideoResult(frames=cached_final["videos"], default_fps=fps)
            except Exception as e:
                logger.debug("[VideoCache] Final cache read failed (%s); running diffusion.", e)

        # ── Hook 1: Text Encoder Cache Interception ───────────────────────────
        if hasattr(underlying, "_blocks") and hasattr(underlying._blocks, "sub_blocks"):
            te_step = underlying._blocks.sub_blocks.get("text_encoder")
            if te_step is not None and video_cache is not None:
                cached_embeds = video_cache.load_embeds()
                if cached_embeds is not None:
                    logger.info("[VideoCache] Prompt embeds cache HIT — skipping UMT5 text encoder pass (24 blocks skipped)!")
                    class WanAnimateTextEncoderHitWrapper:
                        def __init__(self, orig_block, cached_data):
                            self.orig_block = orig_block
                            self.cached_data = cached_data
                        def __getattr__(self, name):
                            return getattr(self.orig_block, name)
                        def __call__(self, components, state):
                            dev = components._execution_device
                            prompt_embeds = self.cached_data["prompt_embeds"].to(dev)
                            prompt_ref_embeds = self.cached_data["prompt_ref_embeds"].to(dev)
                            neg_embeds = self.cached_data.get("negative_prompt_embeds")
                            if neg_embeds is not None:
                                neg_embeds = neg_embeds.to(dev)

                            state.set("prompt_embeds", prompt_embeds)
                            state.set("prompt_ref_embeds", prompt_ref_embeds)
                            state.set("negative_prompt_embeds", neg_embeds)

                            te = getattr(components, "text_encoder", None)
                            if te:
                                evict_module(getattr(te, "model", te))
                            clean_memory(dev)
                            logger.info("[VideoCache] Prompt embeds restored from cache & text encoder evicted.")
                            return components, state

                    underlying._blocks.sub_blocks["text_encoder"] = WanAnimateTextEncoderHitWrapper(te_step, cached_embeds)
                else:
                    logger.info("[VideoCache] Prompt embeds cache MISS — running UMT5 encoder and persisting to cache.")
                    class WanAnimateTextEncoderMissWrapper:
                        def __init__(self, orig_block, v_cache):
                            self.orig_block = orig_block
                            self.v_cache = v_cache
                        def __getattr__(self, name):
                            return getattr(self.orig_block, name)
                        def __call__(self, components, state):
                            components, state = self.orig_block(components, state)
                            prompt_embeds = state.get("prompt_embeds")
                            prompt_ref_embeds = state.get("prompt_ref_embeds")
                            neg_embeds = state.get("negative_prompt_embeds")

                            save_dict = {}
                            if prompt_embeds is not None:
                                save_dict["prompt_embeds"] = prompt_embeds.cpu()
                            if prompt_ref_embeds is not None:
                                save_dict["prompt_ref_embeds"] = prompt_ref_embeds.cpu()
                            if neg_embeds is not None:
                                save_dict["negative_prompt_embeds"] = neg_embeds.cpu()

                            if save_dict:
                                self.v_cache.save_embeds(save_dict)

                            te = getattr(components, "text_encoder", None)
                            if te:
                                evict_module(getattr(te, "model", te))
                            clean_memory(components._execution_device)
                            logger.info("[VideoCache] Prompt embeds saved to disk & text encoder evicted from VRAM.")
                            return components, state

                    underlying._blocks.sub_blocks["text_encoder"] = WanAnimateTextEncoderMissWrapper(te_step, video_cache)

            # ── Hook 2: Reference VAE & CLIP Conditioning Cache ───────────────
            ref_cond_file = video_cache.run_dir / "ref_cond.pt" if video_cache else None
            img_step = underlying._blocks.sub_blocks.get("image_encoder")
            vae_step = underlying._blocks.sub_blocks.get("vae_encoder")
            if ref_cond_file is not None and ref_cond_file.exists():
                logger.info("[VideoCache] Reference VAE & CLIP conditioning cache HIT — skipping image/video vision encoding!")
                try:
                    cached_cond = torch.load(str(ref_cond_file), map_location="cpu", weights_only=False)
                    class CacheHitImageEncoderWrapper:
                        def __init__(self, orig_b, data):
                            self.orig_b = orig_b
                            self.data = data
                        def __getattr__(self, name): return getattr(self.orig_b, name)
                        def __call__(self, comp, state):
                            dev = comp._execution_device
                            if hasattr(state, "set"):
                                if "encoder_hidden_states_image" in self.data:
                                    state.set("encoder_hidden_states_image", self.data["encoder_hidden_states_image"].to(dev))
                                if "condition_clip_context" in self.data:
                                    state.set("condition_clip_context", self.data["condition_clip_context"].to(dev))
                            clean_memory(dev)
                            logger.info("[VideoCache] Restored cached CLIP vision features.")
                            return comp, state

                    class CacheHitVaeEncoderWrapper:
                        def __init__(self, orig_b, data):
                            self.orig_b = orig_b
                            self.data = data
                        def __getattr__(self, name): return getattr(self.orig_b, name)
                        def __call__(self, comp, state):
                            dev = comp._execution_device
                            if hasattr(state, "set") and "reference_image_latents" in self.data:
                                state.set("reference_image_latents", self.data["reference_image_latents"].to(dev))
                            ie = getattr(comp, "image_encoder", None)
                            if ie: evict_module(getattr(ie, "model", ie))
                            clean_memory(dev)
                            logger.info("[VideoCache] Restored cached reference image VAE latents & image_encoder evicted.")
                            return comp, state

                    if img_step is not None:
                        underlying._blocks.sub_blocks["image_encoder"] = CacheHitImageEncoderWrapper(img_step, cached_cond)
                    if vae_step is not None:
                        underlying._blocks.sub_blocks["vae_encoder"] = CacheHitVaeEncoderWrapper(vae_step, cached_cond)
                except Exception as e:
                    logger.debug("[VideoCache] Failed to load ref_cond.pt (%s), will compute.", e)
            elif ref_cond_file is not None:
                class CacheMissVaeEncoderWrapper:
                    def __init__(self, orig_b, v_cache, r_file):
                        self.orig_b = orig_b
                        self.cache = v_cache
                        self.ref_file = r_file
                    def __getattr__(self, name): return getattr(self.orig_b, name)
                    def __call__(self, comp, state):
                        comp, state = self.orig_b(comp, state)
                        save_cond = {}
                        ref_lat = state.get("reference_image_latents")
                        if ref_lat is not None:
                            save_cond["reference_image_latents"] = ref_lat.cpu()
                        img_emb = state.get("encoder_hidden_states_image")
                        if img_emb is not None:
                            save_cond["encoder_hidden_states_image"] = img_emb.cpu()
                        clip_ctx = state.get("condition_clip_context")
                        if clip_ctx is not None:
                            save_cond["condition_clip_context"] = clip_ctx.cpu()
                        if save_cond:
                            self.cache.run_dir.mkdir(parents=True, exist_ok=True)
                            torch.save(save_cond, str(self.ref_file))
                            logger.info("[VideoCache] Reference VAE & CLIP conditioning saved to cache (%s).", self.ref_file.name)
                        ie = getattr(comp, "image_encoder", None)
                        if ie: evict_module(getattr(ie, "model", ie))
                        clean_memory(comp._execution_device)
                        return comp, state

                if vae_step is not None:
                    underlying._blocks.sub_blocks["vae_encoder"] = CacheMissVaeEncoderWrapper(vae_step, video_cache, ref_cond_file)

            # ── Hook 3: Step-by-Step Denoising Cache & Checkpointing ──────────
            core_denoise = underlying._blocks.sub_blocks.get("denoise")
            if core_denoise is not None and hasattr(core_denoise, "sub_blocks"):
                denoise_wrapper = core_denoise.sub_blocks.get("denoise")
                if denoise_wrapper is not None and hasattr(denoise_wrapper, "sub_blocks"):
                    orig_inner = denoise_wrapper.sub_blocks.get("denoise_inner")
                    if orig_inner is not None and video_cache is not None:
                        class WanAnimateDenoiseInnerCacheWrapper:
                            def __init__(self, orig_step, v_cache):
                                self.orig_step = orig_step
                                self.cache = v_cache
                            def __getattr__(self, name):
                                return getattr(self.orig_step, name)
                            def __call__(self, components, block_state, k: int):
                                resume_step = -1
                                resume_latents = None
                                latest_f = self.cache.latest_step_file()
                                if latest_f is not None:
                                    resume_step = self.cache.step_from_file(latest_f)
                                    resume_data = self.cache.load_step(latest_f)
                                    resume_latents = resume_data.get("latents")
                                    if resume_latents is not None:
                                        logger.info("[VideoCache] Found checkpoint %s — resuming from step %d!", latest_f.name, resume_step + 1)

                                # Offload Pass-1 artifacts from VRAM before starting denoise steps.
                                # Do NOT evict the VAE (evict_module moves params to meta device,
                                # destroying them; decode would then fail with "Tensor on meta").
                                if hasattr(block_state, "driving_video_latents") and block_state.driving_video_latents is not None:
                                    block_state.driving_video_latents = [
                                        l.cpu() if isinstance(l, torch.Tensor) else l
                                        for l in block_state.driving_video_latents
                                    ]
                                clean_memory(components._execution_device)
                                logger.info("[VideoCache] Offloaded driving video latents to CPU; cleaned VRAM for denoising.")

                                transformer_dtype = components.transformer.dtype
                                guider_inputs = {
                                    "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
                                    "is_uncondtion": (False, True),
                                }
                                import inspect
                                transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
                                shared_kwargs = {
                                    name: value.to(transformer_dtype) if isinstance(value, torch.Tensor) else value
                                    for name, value in block_state.denoiser_input_fields.items()
                                    if name in transformer_args and name not in guider_inputs
                                }

                                # Pre-cast tensors that are fixed for the entire segment to
                                # avoid redundant .to() allocations on every denoising step.
                                _dev = components.transformer.device if hasattr(components.transformer, "device") else block_state.latents.device
                                _reference_latents = block_state.reference_latents.to(dtype=transformer_dtype, device=_dev)
                                _prompt_embeds = block_state.prompt_embeds.to(dtype=transformer_dtype)
                                _neg_embeds = block_state.negative_prompt_embeds.to(dtype=transformer_dtype)
                                guider_inputs_typed = {
                                    "encoder_hidden_states": (_prompt_embeds, _neg_embeds),
                                    "is_uncondtion": (False, True),
                                }

                                total_steps = len(block_state.timesteps)
                                from tqdm import tqdm
                                with tqdm(total=total_steps, desc=f"Segment {k + 1}/{block_state.num_segments}") as progress_bar:
                                    for i, t in enumerate(block_state.timesteps):
                                        if resume_step >= 0 and i < resume_step:
                                            logger.info("    [VideoCache] Fast-forwarding step %d/%d (completed in prior run)...", i + 1, total_steps)
                                            progress_bar.update()
                                            continue
                                        elif resume_step >= 0 and i == resume_step and resume_latents is not None:
                                            logger.info("    [VideoCache] Injected saved latents at step %d/%d.", i + 1, total_steps)
                                            block_state.latents = resume_latents.to(device=block_state.latents.device, dtype=block_state.latents.dtype)
                                            progress_bar.update()
                                            continue

                                        timestep = torch.stack([t])
                                        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
                                        guider_state = components.guider.prepare_inputs(guider_inputs_typed)

                                        # Cast latents to transformer dtype once per step (latents update each step)
                                        _latents_typed = block_state.latents.to(transformer_dtype)

                                        for guider_state_batch in guider_state:
                                            components.guider.prepare_models(components.transformer)
                                            guider_state_batch.noise_pred = components.transformer(
                                                [_latents_typed],
                                                timestep=timestep,
                                                encoder_hidden_states=[guider_state_batch.encoder_hidden_states[0]],
                                                condition_latents=[_reference_latents],
                                                kv_cache=block_state.kv_cache,
                                                kv_cache_mode="cached",
                                                seq_len=block_state.max_seq_len,
                                                reference_grid_sizes=block_state.grid_sizes_ref,
                                                origin_len=block_state.segment_frame_length,
                                                origin_area=[block_state.height, block_state.width],
                                                is_uncondtion=guider_state_batch.is_uncondtion,
                                                **shared_kwargs,
                                            ).sample[0]
                                            components.guider.cleanup_models(components.transformer)

                                        noise_pred = components.guider(guider_state)[0]
                                        latents = components.scheduler.step(
                                            noise_pred.unsqueeze(0),
                                            t,
                                            block_state.latents.unsqueeze(0),
                                            return_dict=False,
                                            generator=block_state.generator,
                                        )[0]
                                        block_state.latents = latents.squeeze(0)

                                        # Checkpoint step
                                        if (i + 1) % self.cache.save_every == 0 or (i + 1) == total_steps:
                                            step_path = self.cache.run_dir / f"step_{i:04d}.pt"
                                            torch.save({"latents": block_state.latents.cpu()}, str(step_path))
                                            logger.info("    [VideoCache] Step %d/%d checkpoint saved (%s)", i + 1, total_steps, step_path.name)

                                        progress_bar.update()

                                return components, block_state

                        denoise_wrapper.sub_blocks["denoise_inner"] = WanAnimateDenoiseInnerCacheWrapper(orig_inner, video_cache)

                    # ── Hook 4: Transformer Eviction Before VAE Decode ────────
                    orig_decode = denoise_wrapper.sub_blocks.get("decode")
                    if orig_decode is not None:
                        class WanAnimateSegmentDecodeWrapper:
                            def __init__(self, orig_dec):
                                self.orig_dec = orig_dec
                            def __getattr__(self, name):
                                return getattr(self.orig_dec, name)
                            def __call__(self, comp, b_state, k: int):
                                tr = getattr(comp, "transformer", None)
                                if tr is not None:
                                    evict_module(getattr(tr, "model", tr))
                                clean_memory(comp._execution_device)
                                logger.info("[VideoCache] Transformer evicted from VRAM before VAE decode.")

                                comp, b_state = self.orig_dec(comp, b_state, k=k)
                                logger.info("[VideoCache] Segment %d VAE decode complete.", k + 1)
                                return comp, b_state

                        denoise_wrapper.sub_blocks["decode"] = WanAnimateSegmentDecodeWrapper(orig_decode)

        # Run generation through underlying modular pipeline
        out = underlying(
            image=first_frame,
            driving_video=driving_video,
            driving_video_fps=driving_video_fps,
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=target_h,
            width=target_w,
            fps=fps,
            segment_frame_length=segment_frame_length,
            prev_segment_conditioning_frames=prev_segment_conditioning_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        # Extract resulting video frames
        frames = getattr(out, "frames", None)
        if frames is None and hasattr(out, "videos"):
            frames = out.videos
        elif frames is None and hasattr(out, "get"):
            frames = out.get("videos", out.get("frames", None))

        # ── Save Final Output Cache ───────────────────────────────────────────
        if video_cache is not None and frames is not None:
            try:
                video_cache.save_final({"videos": frames})
                logger.info("[VideoCache] Final video frames saved to cache (%s/final.pt)", video_cache.run_dir.name)
            except Exception as e:
                logger.debug("[VideoCache] Could not save final cache: %s", e)

        return WeeVideoResult(frames=frames, default_fps=fps)

