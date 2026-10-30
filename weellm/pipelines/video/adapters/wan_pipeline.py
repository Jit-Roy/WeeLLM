"""
weellm.pipelines.video.adapters.wan_pipeline
============================================
Custom adapter for Wan video diffusion pipelines (Wan2.1 / Wan2.2).
Handles text-to-video and image-to-video modes, unnormalization of latent
variables, and default generation settings.
"""

from __future__ import annotations

import logging
import torch
from typing import Optional

from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline, WeeVideoResult

logger = logging.getLogger("weellm")

# Monkey-patch Diffusers Wan I2V basic_clean using standard library html only (no ftfy)
try:
    import html
    import diffusers.pipelines.wan.pipeline_wan_i2v as _i2v_mod

    def _safe_basic_clean(text):
        return html.unescape(html.unescape(text)).strip()

    _i2v_mod.basic_clean = _safe_basic_clean
except Exception:
    pass

# Default Wan motion-inducing negative prompt (mandated by official Wan2.2 training)
DEFAULT_WAN_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, "
    "static, still image, motionless, frozen, blurry, ugly, low quality"
)


class WeeWanPipeline(WeeVideoPipeline):
    """
    WeeLLM Pipeline adapter for Wan video generation models (e.g. Wan2.2-TI2V-5B).
    Inherits from WeeVideoPipeline and seamlessly supports both text-to-video
    and image-to-video tasks with layer-streaming memory optimization.
    """

    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        self._i2v_pipeline = None

        # Wan VAE: native bfloat16 if GPU supports it, else float32
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        vae = getattr(pipeline, "vae", None)
        if vae is not None and getattr(vae, "dtype", None) != target_dtype:
            vae.to(dtype=target_dtype)
        vae_streamer = getattr(pipeline, "vae_streamer", None)
        if vae_streamer is not None:
            vae_streamer.dtype = target_dtype

    def _preprocess_latents_for_decode(self, latents: torch.Tensor, vae, kwargs):
        """
        Unnormalize latents using Wan's specific latents_mean and latents_std
        before passing into AutoencoderKLWan.decode().
        """
        if hasattr(vae, "config") and hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            z_dim = getattr(vae.config, "z_dim", latents.shape[1] if latents.dim() >= 2 else 48)
            mean = (
                torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            # diffusers Wan unnormalization:
            # latents_std_inv = 1.0 / latents_std
            # latents = latents / latents_std_inv + mean  ==>  latents = latents * latents_std + mean
            std = (
                torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents = latents * std + mean
            return latents

        return super()._preprocess_latents_for_decode(latents, vae, kwargs)

    def _snap_dims(self, kwargs: dict, first_frame) -> None:
        """
        Snap height/width to Wan's 32-multiple requirement.
        Follows the robust dimension resolution pattern from MiniMax H3.
        """
        import PIL.Image
        _h, _w = kwargs.get("height"), kwargs.get("width")
        _img = first_frame

        # 1. Derive missing dimensions from first_frame if present
        if _h is None or _w is None:
            h, w = _h, _w
            _first = _img[0] if isinstance(_img, list) and _img else _img

            if isinstance(_first, PIL.Image.Image):
                w, h = w or _first.width, h or _first.height
            elif hasattr(_first, "shape") and len(_first.shape) >= 2:
                h, w = h or int(_first.shape[-2]), w or int(_first.shape[-1])

            kwargs["height"] = max(32, round((h or 512) / 32) * 32)
            kwargs["width"] = max(32, round((w or 512) / 32) * 32)

            if kwargs["height"] != h or kwargs["width"] != w:
                logger.info("[WeeLLM/Wan] Snapped dims %sx%s -> %dx%d", str(w), str(h), kwargs["width"], kwargs["height"])
        else:
            kwargs["height"] = max(32, round(int(_h) / 32) * 32)
            kwargs["width"] = max(32, round(int(_w) / 32) * 32)

    def __call__(self, **kwargs):
        # 1. Resolve first_frame and last_frame.
        # Accept either 'first_frame' or 'image' as the conditioning frame.
        # 'image' is treated as an alias for 'first_frame'.
        first_frame = kwargs.pop("first_frame", None)
        if first_frame is None:
            first_frame = kwargs.pop("image", None)
        else:
            kwargs.pop("image", None)  

        last_frame = kwargs.pop("last_frame", None)
        def _load_frame(f):
            if f is None:
                return None
            if isinstance(f, str):
                from PIL import Image, ImageOps
                return ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            if isinstance(f, list) and len(f) > 0 and isinstance(f[0], str):
                from PIL import Image, ImageOps
                return [ImageOps.exif_transpose(Image.open(x)).convert("RGB") for x in f]
            return f

        first_frame = _load_frame(first_frame)
        last_frame = _load_frame(last_frame)

        # 2. Wan dimensions and framing constraints
        self._snap_dims(kwargs, first_frame)
        target_w, target_h = kwargs["width"], kwargs["height"]

        # Physically resize first_frame and last_frame to target dimensions (MiniMax H3 pattern)
        import PIL.Image
        if isinstance(first_frame, PIL.Image.Image) and first_frame.size != (target_w, target_h):
            first_frame = first_frame.resize((target_w, target_h), PIL.Image.Resampling.LANCZOS)
        elif isinstance(first_frame, list) and len(first_frame) > 0 and isinstance(first_frame[0], PIL.Image.Image):
            first_frame = [i.resize((target_w, target_h), PIL.Image.Resampling.LANCZOS) for i in first_frame]

        if isinstance(last_frame, PIL.Image.Image) and last_frame.size != (target_w, target_h):
            last_frame = last_frame.resize((target_w, target_h), PIL.Image.Resampling.LANCZOS)
        elif isinstance(last_frame, list) and len(last_frame) > 0 and isinstance(last_frame[0], PIL.Image.Image):
            last_frame = [i.resize((target_w, target_h), PIL.Image.Resampling.LANCZOS) for i in last_frame]

        if first_frame is not None:
            kwargs["image"] = first_frame
        if last_frame is not None:
            kwargs["last_image"] = last_frame

        # Temporal frames must satisfy (num_frames - 1) % 4 == 0
        if kwargs.get("num_frames") is not None:
            nf = int(kwargs["num_frames"])
            if nf > 1:
                kwargs["num_frames"] = max(1, ((nf - 1) // 4) * 4 + 1)

        # 3. Default settings
        if first_frame is None:
            # T2V defaults (720P)
            kwargs.setdefault("height", 704)
            kwargs.setdefault("width", 1280)

        kwargs.setdefault("num_frames", 81)
        kwargs.setdefault("num_inference_steps", 50)
        kwargs.setdefault("guidance_scale", 5.0)
        kwargs.setdefault("max_sequence_length", 512)
        kwargs.setdefault("fps", 24)

        # 4. If first_frame conditioning is active, route through WanImageToVideoPipeline
        #
        # IMPORTANT: We must NOT force output_type="latent" for the I2V path.
        # WanImageToVideoPipeline (with expand_timesteps=True) applies a critical
        # first-frame compositing step INSIDE the pipeline just before VAE decode:
        #
        #   latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        #
        # This step is SKIPPED when output_type="latent" (it is guarded by
        # "if not output_type == 'latent'"), which causes the first-frame latents
        # to contain unblended noise from denoising, producing corrupted output
        # (blue flickers, broken animation, incomplete frames).
        #
        # Solution: Let the diffusers I2V pipeline handle its own decode
        # (output_type="np"), then wrap the resulting numpy frames ourselves.
        if first_frame is not None:
            underlying = getattr(self, "_pipeline", self)
            if self._i2v_pipeline is None:
                from diffusers import WanImageToVideoPipeline
                self._i2v_pipeline = WanImageToVideoPipeline(
                    tokenizer=underlying.tokenizer,
                    text_encoder=underlying.text_encoder,
                    transformer=underlying.transformer,
                    vae=underlying.vae,
                    scheduler=underlying.scheduler,
                    expand_timesteps=getattr(underlying.config, "expand_timesteps", True),
                )
                self._i2v_pipeline.model_dir = getattr(underlying, "model_dir", "")
                real_dev = getattr(underlying, "device", torch.device("cuda"))
                if not hasattr(WanImageToVideoPipeline, "_weellm_patched_dev"):
                    WanImageToVideoPipeline._execution_device = property(lambda self_obj: real_dev)
                    WanImageToVideoPipeline.device = property(lambda self_obj: real_dev)
                    WanImageToVideoPipeline._weellm_patched_dev = True
                logger.info("[WeeLLM/Wan] Built WanImageToVideoPipeline for I2V mode (first_frame%s).",
                            " + last_frame" if last_frame is not None else "")

            else:
                real_dev = getattr(underlying, "device", torch.device("cuda"))

            # Extract cache settings
            fps = kwargs.pop("fps", 24)
            fresh = kwargs.pop("fresh", False)
            no_cache = kwargs.pop("no_cache", False)
            cache_every = kwargs.pop("cache_every", 1)
            lora_weights = kwargs.pop("lora_weights", None)
            kwargs.pop("latent_upsampler", None)
            generator = kwargs.get("generator")
            seed = kwargs.pop("seed", generator.initial_seed() if generator is not None else 42)
            i2v_flow_shift = kwargs.pop("i2v_flow_shift", None)

            prompt = kwargs.get("prompt", "")
            height = kwargs.get("height", 512)
            width = kwargs.get("width", 512)
            num_frames = kwargs.get("num_frames", 81)
            steps = kwargs.get("num_inference_steps", 10)

            neg_prompt = kwargs.get("negative_prompt") or DEFAULT_WAN_NEG_PROMPT

            _video_cache = None
            if not no_cache:
                _video_cache = self._setup_cache(
                    prompt, height, width, num_frames, steps, seed,
                    negative_prompt=neg_prompt,
                    save_every=cache_every, lora_weights=lora_weights
                )
                if fresh:
                    _video_cache.clear_run()

            # Fast path: decode-only cache hit
            if _video_cache is not None and _video_cache.has_final():
                logger.info("[VideoCache] Final-latents cache HIT — running decode-only.")
                _cached_final = _video_cache.load_final()
                _lat = _cached_final.get("latents")
                if _lat is not None:
                    with torch.no_grad():
                        from weellm.io.memory import evict_module
                        import gc
                        # Evict transformer and text encoder to free VRAM for VAE decode.
                        # IMPORTANT: underlying.transformer is a WeeLLM streamer (not nn.Module).
                        # evict_module needs the inner .model to find resident tensors on CUDA.
                        for _tr_name in ("transformer", "text_encoder", "text_encoder_2"):
                            _tr = getattr(underlying, _tr_name, None)
                            if _tr is not None:
                                # Resolve inner nn.Module if this is a WeeLLM streamer wrapper
                                _inner = getattr(_tr, "model", _tr)
                                n_evicted = evict_module(_inner)
                                logger.info("[WeeLLM/Wan] Cache-hit evict '%s': %d tensors freed.",
                                            _tr_name, n_evicted)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        vae = self._i2v_pipeline.vae
                        if getattr(self._i2v_pipeline.config, "expand_timesteps", True):
                            import sys
                            logger.info("[WeeLLM/Wan] Step 1/3: Preprocessing first frame (%dx%d)...", width, height)
                            sys.stdout.flush()
                            image_tensor = self._i2v_pipeline.video_processor.preprocess(
                                first_frame, height=height, width=width
                            ).to(real_dev, dtype=torch.float32)
                            last_image_tensor = None
                            if last_frame is not None:
                                last_image_tensor = self._i2v_pipeline.video_processor.preprocess(
                                    last_frame, height=height, width=width
                                ).to(real_dev, dtype=torch.float32)
                            num_channels_latents = vae.config.z_dim
                            logger.info("[WeeLLM/Wan] Step 2/3: Encoding first-frame condition with VAE...",)
                            sys.stdout.flush()
                            latents_outputs = self._i2v_pipeline.prepare_latents(
                                image=image_tensor,
                                batch_size=1,
                                num_channels_latents=num_channels_latents,
                                height=height,
                                width=width,
                                num_frames=num_frames,
                                dtype=torch.float32,
                                device=real_dev,
                                generator=None,
                                latents=None,
                                last_image=last_image_tensor,
                            )
                            _, condition, first_frame_mask = latents_outputs
                            logger.info("[WeeLLM/Wan] Step 3/3: Blending clean first-frame condition into cached latents at frame 0...")
                            sys.stdout.flush()
                            _lat = (1 - first_frame_mask) * condition + first_frame_mask * _lat.to(real_dev, dtype=torch.float32)
                            del image_tensor, last_image_tensor, condition, first_frame_mask, latents_outputs
                            from weellm.io.utils import clean_memory
                            clean_memory(real_dev)

                        _lat = _lat.to(device=real_dev, dtype=vae.dtype)
                        logger.info("[WeeLLM/Wan] Denormalizing latents (mean + std)...")
                        sys.stdout.flush()
                        latents_mean = (
                            torch.tensor(vae.config.latents_mean)
                            .view(1, vae.config.z_dim, 1, 1, 1)
                            .to(_lat.device, _lat.dtype)
                        )
                        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                            _lat.device, _lat.dtype
                        )
                        _lat = _lat / latents_std + latents_mean
                        # Explicit GC before decoder to reclaim any VAE encoder allocations
                        from weellm.io.utils import clean_memory
                        clean_memory(real_dev)
                        import sys
                        logger.info("[WeeLLM/Wan] Decoding video from latents (shape: %s | free VRAM: %.1f MB)...",
                                    tuple(_lat.shape),
                                    (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e6 if torch.cuda.is_available() else 0.0)
                        sys.stdout.flush()
                        video = vae.decode(_lat, return_dict=False)[0]
                        logger.info(
                            "[WeeLLM/Wan] VAE decode finished (output shape: %s). Postprocessing frames...",
                            tuple(video.shape) if hasattr(video, "shape") else type(video)
                        )
                        sys.stdout.flush()
                        video = self._i2v_pipeline.video_processor.postprocess_video(video, output_type="pil")
                        logger.info("[WeeLLM/Wan] Postprocessing complete.")
                        sys.stdout.flush()

                    frames = video[0] if isinstance(video, list) and len(video) > 0 else video
                    return WeeVideoResult(frames=frames, default_fps=fps)

            # Normal path: full I2V diffusion run
            i2v_kwargs = dict(kwargs)
            i2v_kwargs.setdefault("output_type", "pil")
            if i2v_kwargs.get("generator") is None and seed is not None:
                real_dev = getattr(underlying, "device", torch.device("cuda"))
                i2v_kwargs["generator"] = torch.Generator(device=real_dev).manual_seed(int(seed))

            if _video_cache is not None:
                _cache_cb = _video_cache.get_step_callback()
                _user_cb = i2v_kwargs.get("callback_on_step_end")
                if _user_cb is not None:
                    # Chain: run cache callback first, then user callback
                    def _chained_cb(pipe_obj, step_index, timestep, callback_kwargs, _c=_cache_cb, _u=_user_cb):
                        callback_kwargs = _c(pipe_obj, step_index, timestep, callback_kwargs) or callback_kwargs
                        callback_kwargs = _u(pipe_obj, step_index, timestep, callback_kwargs) or callback_kwargs
                        return callback_kwargs
                    i2v_kwargs["callback_on_step_end"] = _chained_cb
                else:
                    i2v_kwargs["callback_on_step_end"] = _cache_cb
                i2v_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

                # Intercept encode_prompt for prompt embeddings caching
                _video_cache.wrap_pipeline_encode_prompt(self._i2v_pipeline, device=str(real_dev))

            i2v_kwargs["negative_prompt"] = neg_prompt
            if not kwargs.get("negative_prompt"):
                logger.info("[WeeLLM/Wan] Injected official Wan motion-inducing negative prompt.")

            logger.info("[WeeLLM/Wan] Running I2V: %d frames, %dx%d, output_type=%s, flow_shift=%s",
                        i2v_kwargs.get("num_frames", 81),
                        i2v_kwargs.get("width", 0),
                        i2v_kwargs.get("height", 0),
                        i2v_kwargs["output_type"],
                        str(i2v_flow_shift if i2v_flow_shift is not None else getattr(self._i2v_pipeline.scheduler.config, 'flow_shift', 5.0)))

            # Only override flow_shift if user explicitly passed i2v_flow_shift
            _scheduler = self._i2v_pipeline.scheduler
            _orig_flow_shift = getattr(_scheduler.config, "flow_shift", None)
            _patched_flow = False
            if i2v_flow_shift is not None and _orig_flow_shift != i2v_flow_shift:
                try:
                    from collections import OrderedDict
                    OrderedDict.__setitem__(_scheduler.config, "flow_shift", i2v_flow_shift)
                    object.__setattr__(_scheduler.config, "flow_shift", i2v_flow_shift)
                    _patched_flow = True
                    logger.info(
                        "[WeeLLM/Wan] Patched scheduler flow_shift: %.1f -> %.1f (explicit override)",
                        _orig_flow_shift, i2v_flow_shift
                    )
                except Exception as _e:
                    logger.warning("[WeeLLM/Wan] Could not patch flow_shift (%s), running with config default.", _e)

            try:
                with torch.no_grad():
                    out = self._i2v_pipeline(**i2v_kwargs)
            finally:
                # Restore original flow_shift if explicitly overridden
                if _patched_flow and _orig_flow_shift is not None:
                    try:
                        from collections import OrderedDict
                        OrderedDict.__setitem__(_scheduler.config, "flow_shift", _orig_flow_shift)
                        object.__setattr__(_scheduler.config, "flow_shift", _orig_flow_shift)
                    except Exception:
                        pass

            frames = getattr(out, "frames", None)
            while isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
                frames = frames[0]

            if _video_cache is not None:
                last_step_file = _video_cache.run_dir / f"step_{steps-1:04d}.pt"
                if last_step_file.exists():
                    try:
                        step_data = torch.load(str(last_step_file), map_location="cpu", weights_only=False)
                        _video_cache.save_final(step_data)
                    except Exception as e:
                        logger.debug(f"[VideoCache] Failed to save final from step: {e}")

            return WeeVideoResult(
                frames=frames,
                default_fps=fps,
            )

        # T2V: use normal parent call (streaming transformer + manual decode).
        # WanPipeline.__call__ does not accept 'fps' — pop it before delegating.
        kwargs.pop("fps", None)
        kwargs.pop("frame_rate", None)
        if not kwargs.get("negative_prompt"):
            kwargs["negative_prompt"] = DEFAULT_WAN_NEG_PROMPT
            logger.info("[WeeLLM/Wan] Injected official Wan motion-inducing negative prompt for T2V.")
        return super().__call__(**kwargs)
