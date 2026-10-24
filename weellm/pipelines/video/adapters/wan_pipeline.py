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


class WeeWanPipeline(WeeVideoPipeline):
    """
    WeeLLM Pipeline adapter for Wan video generation models (e.g. Wan2.2-TI2V-5B).
    Inherits from WeeVideoPipeline and seamlessly supports both text-to-video
    and image-to-video tasks with layer-streaming memory optimization.
    """

    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        self._i2v_pipeline = None

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

    def _snap_dims(self, kwargs: dict, input_image) -> None:
        """Snap height/width to Wan's 32-multiple requirement."""
        if kwargs.get("height") is not None:
            kwargs["height"] = max(32, (int(kwargs["height"]) // 32) * 32)
        if kwargs.get("width") is not None:
            kwargs["width"] = max(32, (int(kwargs["width"]) // 32) * 32)

        # Derive from image if dims not provided
        if input_image is not None:
            if "width" not in kwargs or kwargs["width"] is None:
                kwargs["width"] = max(32, (input_image.width // 32) * 32)
            if "height" not in kwargs or kwargs["height"] is None:
                kwargs["height"] = max(32, (input_image.height // 32) * 32)

    def __call__(self, **kwargs):
        # 1. Handle input image for Image-to-Video if provided
        input_image = kwargs.get("image", None)
        if input_image is not None and isinstance(input_image, str):
            from PIL import Image, ImageOps
            input_image = ImageOps.exif_transpose(Image.open(input_image)).convert("RGB")
            kwargs["image"] = input_image

        # 2. Wan dimensions and framing constraints
        self._snap_dims(kwargs, input_image)

        # Temporal frames must satisfy (num_frames - 1) % 4 == 0
        if kwargs.get("num_frames") is not None:
            nf = int(kwargs["num_frames"])
            if nf > 1:
                kwargs["num_frames"] = max(1, ((nf - 1) // 4) * 4 + 1)

        # 3. Default settings
        if input_image is None:
            # T2V defaults (720P)
            kwargs.setdefault("height", 704)
            kwargs.setdefault("width", 1280)

        kwargs.setdefault("num_frames", 81)
        kwargs.setdefault("num_inference_steps", 50)
        kwargs.setdefault("guidance_scale", 5.0)
        kwargs.setdefault("max_sequence_length", 512)
        kwargs.setdefault("fps", 24)

        # 4. If image conditioning is active, route through WanImageToVideoPipeline
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
        if input_image is not None:
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
                logger.info("[WeeLLM/Wan] Built WanImageToVideoPipeline for I2V mode.")

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

            prompt = kwargs.get("prompt", "")
            height = kwargs.get("height", 512)
            width = kwargs.get("width", 512)
            num_frames = kwargs.get("num_frames", 81)
            steps = kwargs.get("num_inference_steps", 10)

            _video_cache = None
            if not no_cache:
                _video_cache = self._setup_cache(
                    prompt, height, width, num_frames, steps, seed,
                    negative_prompt=kwargs.get("negative_prompt"),
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
                    from weellm.io.memory import evict_module
                    for _tr_name in ("transformer", "text_encoder", "text_encoder_2"):
                        _tr = getattr(underlying, _tr_name, None)
                        if _tr is not None:
                            evict_module(_tr)
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    vae = self._i2v_pipeline.vae
                    if getattr(self._i2v_pipeline.config, "expand_timesteps", True):
                        image_tensor = self._i2v_pipeline.video_processor.preprocess(
                            input_image, height=height, width=width
                        ).to(real_dev, dtype=torch.float32)
                        num_channels_latents = vae.config.z_dim
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
                        )
                        _, condition, first_frame_mask = latents_outputs
                        _lat = (1 - first_frame_mask) * condition + first_frame_mask * _lat.to(real_dev, dtype=torch.float32)
                        del image_tensor, condition, first_frame_mask, latents_outputs
                        from weellm.io.utils import clean_memory
                        clean_memory(real_dev)

                    _lat = _lat.to(device=real_dev, dtype=vae.dtype)
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(_lat.device, _lat.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                        _lat.device, _lat.dtype
                    )
                    _lat = _lat / latents_std + latents_mean
                    import sys
                    logger.info("[WeeLLM/Wan] Decoding video from latents (shape: %s)...", tuple(_lat.shape))
                    sys.stdout.flush()
                    with torch.no_grad():
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
            if _video_cache is not None:
                i2v_kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
                i2v_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            logger.info("[WeeLLM/Wan] Running I2V: %d frames, %dx%d, output_type=%s",
                        i2v_kwargs.get("num_frames", 81),
                        i2v_kwargs.get("width", 0),
                        i2v_kwargs.get("height", 0),
                        i2v_kwargs["output_type"])

            with torch.no_grad():
                out = self._i2v_pipeline(**i2v_kwargs)

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
        return super().__call__(**kwargs)
