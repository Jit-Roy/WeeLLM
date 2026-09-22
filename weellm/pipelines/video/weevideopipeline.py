"""
Generic Video Pipeline wrapper featuring auto-routing for video models
and a unified video cache.
"""

import os
import gc
import json
import torch
import inspect
import logging
import importlib

from weellm.pipelines.weebasepipeline import WeeBasePipeline

logger = logging.getLogger("weellm")

# ---------------------------------------------------------------------------
# Pipeline class name → (module path, class name)
# ---------------------------------------------------------------------------
_VIDEO_MAP = {
    "LTX2Pipeline":("weellm.pipelines.video.adapters.ltx2_pipeline","WeeLTX2Pipeline"),
    "MiniMaxH3ModularPipeline":("weellm.pipelines.video.adapters.minimax_h3_modular_pipeline","WeeMiniMaxPipeline"),
    "WanPipeline":("weellm.pipelines.video.weevideopipeline","WeeVideoPipeline"),
    "CogVideoXPipeline":("weellm.pipelines.video.weevideopipeline","WeeVideoPipeline"),
}

class WeeVideoResult:
    """
    A unified wrapper for generated video frames and audio (if any),
    mimicking the PIL.Image experience for video generation.
    """
    def __init__(self, frames, audio=None, default_fps=24, audio_rate=48000):
        # Un-nest if it's a list of lists (e.g. batch size 1 list of frames)
        while isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
            frames = frames[0]
            
        # Unbatch if it's a 5D tensor [B, C, F, H, W] -> [C, F, H, W]
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 5:
                frames = frames[0]
            # Convert [C, F, H, W] to list of numpy arrays [F, H, W, C] for export
            if frames.dim() == 4:
                import numpy as np
                frames = frames.permute(1, 2, 3, 0).cpu().float().numpy()
                frames = [np.clip(f * 255.0, 0, 255).astype(np.uint8) if f.max() <= 1.0 else f for f in frames]
                
        self.frames = frames
        self.audio = audio
        self.default_fps = default_fps
        self.audio_rate = audio_rate

    def save(self, filepath: str, fps: int = None, **kwargs):
        final_fps = fps if fps is not None else self.default_fps
        
        try:
            from diffusers.utils import encode_video
            has_encode = True
        except ImportError:
            has_encode = False
            
        if has_encode and self.audio is not None:
            # Format audio correctly (typically requires float CPU tensor)
            _audio = self.audio
            _valid_audio = False
            
            if isinstance(_audio, torch.Tensor):
                _audio = _audio.float().cpu()
                # Unbatch if shape is [1, channels, samples]
                if _audio.dim() == 3 and _audio.shape[0] == 1:
                    _audio = _audio[0]
                    
                # A decoded audio waveform should be [channels, samples] where channels is 1 or 2
                if _audio.dim() == 2 and _audio.shape[0] in [1, 2]:
                    _valid_audio = True
                elif _audio.dim() == 1:
                    _valid_audio = True
                    _audio = _audio.unsqueeze(0)
            
            if _valid_audio:
                encode_video(
                    self.frames,
                    fps=final_fps,
                    output_path=filepath,
                    audio=_audio,
                    audio_sample_rate=self.audio_rate,
                    **kwargs
                )
            else:
                logger.warning(f"Audio tensor has invalid shape for encoding: {getattr(self.audio, 'shape', 'unknown')}. Exporting video without audio.")
                from diffusers.utils import export_to_video
                export_to_video(self.frames, output_video_path=filepath, fps=final_fps)
        else:
            from diffusers.utils import export_to_video
            export_to_video(self.frames, output_video_path=filepath, fps=final_fps)
            
        logger.info(f"Video saved to {filepath} at {final_fps} FPS.")

class WeeVideoPipeline(WeeBasePipeline):
    
    @classmethod
    def _get_diffusers_pipeline_class(cls, index: dict) -> str:
        pipeline_class_name = index.get("_class_name")
        if not pipeline_class_name:
            raise ValueError("No _class_name found in model_index.json")
        return pipeline_class_name

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        index_path = os.path.join(model_dir, "model_index.json")
        class_name = ""
        
        if not os.path.exists(index_path):
            try:
                from huggingface_hub import hf_hub_download
                index_path = hf_hub_download(model_dir, "model_index.json")
            except Exception:
                pass
                
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                class_name = json.load(f).get("_class_name", "")
                
        # If this method is called strictly on WeeVideoPipeline (the base), do the routing.
        # But if it's called on a subclass directly (e.g. WeeLTX2Pipeline.from_pretrained),
        # we skip the routing to avoid infinite loops and just construct it.
        if cls is WeeVideoPipeline and class_name in _VIDEO_MAP:
            module_path, class_name_str = _VIDEO_MAP[class_name]
            logger.info("  Mode:     %s", class_name)

            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, class_name_str)
            return pipeline_class.from_pretrained(model_dir, **kwargs)
            
        return super().from_pretrained(model_dir, **kwargs)

    def _preprocess_latents_for_decode(self, latents, vae, kwargs):
        """
        Hook for model-specific latent preprocessing before VAE decode.
        Base implementation handles standard diffusers scaling/shifting.
        """
        if hasattr(vae, "config"):
            _scaling = getattr(vae.config, "scaling_factor", 1.0)
            _shift = getattr(vae.config, "shift_factor", 0.0)
            
            if _scaling != 1.0 or _shift != 0.0:
                latents = (latents / _scaling) - _shift
                
        return latents

    def _setup_cache(self, prompt, height, width, num_frames, steps, seed, save_every=1, cache_root=None, lora_weights=None):
        from weellm.pipelines.video.video_cache import VideoStepCache
        if cache_root is None:
            cache_root = os.path.join(
                os.path.dirname(os.path.abspath(self.model_dir)), ".weellm_cache"
            )
        cache = VideoStepCache(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            steps=steps,
            seed=seed,
            lora_weights=lora_weights,
            save_every=save_every,
            cache_root=cache_root,
        )
        os.makedirs(cache.run_dir_path, exist_ok=True)
        return cache

    def __call__(self, **kwargs):
        """
        Generic video generation loop featuring VideoStepCache integration,
        text encoder eviction on cache hits, and manual VAE decoding.
        """
        no_cache = kwargs.pop("no_cache", False)
        cache_every = kwargs.pop("cache_every", 1)
        fresh = kwargs.pop("fresh", False)
        
        lora_weights = kwargs.pop("lora_weights", None)
        latent_upsampler = kwargs.pop("latent_upsampler", None)
        user_output_type = kwargs.get("output_type", "pil")
        
        generator = kwargs.get("generator")
        seed = kwargs.pop("seed", generator.initial_seed() if generator is not None else 42)
        
        prompt = kwargs.get("prompt", "")
        num_frames = kwargs.get("num_frames", 121)
        steps = kwargs.get("num_inference_steps", 6)

        # ── Smart height/width resolution ─────────────────────────────────────
        # Priority:
        #   1. User explicitly passed both → use as-is, ignore everything else.
        #   2. Image-to-video (kwargs["image"] set by WeeLTX2Pipeline router):
        #        - Missing dim(s) are filled from the first frame's actual size.
        #   3. Text-to-video (no image): default to 544×960.
        # In all cases the resolved values are written back into kwargs so the
        # underlying diffusers pipeline and the cache both see the same dims.
        _user_h = kwargs.get("height")
        _user_w = kwargs.get("width")
        _first_frame = kwargs.get("image")   # set by WeeLTX2Pipeline for I2V

        if _user_h is not None and _user_w is not None:
            # Both provided by the user — respect exactly, don't look at image dims.
            height, width = _user_h, _user_w
        else:
            if _first_frame is not None:
                # I2V mode: derive missing dim(s) from the supplied first frame.
                _fh, _fw = None, None
                try:
                    from PIL import Image as _PILImage
                    if isinstance(_first_frame, _PILImage.Image):
                        _fw, _fh = _first_frame.size   # PIL.size → (width, height)
                    elif isinstance(_first_frame, torch.Tensor):
                        _t = _first_frame
                        if _t.dim() == 4:
                            _t = _t[0]          # unbatch [B,C,H,W] → [C,H,W]
                        if _t.dim() == 3:
                            if _t.shape[0] in (1, 3, 4):   # CHW layout
                                _fh, _fw = int(_t.shape[1]), int(_t.shape[2])
                            else:                            # HWC layout
                                _fh, _fw = int(_t.shape[0]), int(_t.shape[1])
                    else:
                        import numpy as _np
                        _arr = _np.asarray(_first_frame)
                        _fh, _fw = int(_arr.shape[0]), int(_arr.shape[1])
                except Exception as _dim_err:
                    logger.warning(
                        "[WeeLLM] Could not extract dims from first frame (%s) — "
                        "falling back to 544×960.", _dim_err
                    )

                height = _user_h if _user_h is not None else (_fh if _fh else 544)
                width  = _user_w if _user_w is not None else (_fw if _fw else 960)
                logger.info(
                    "[WeeLLM] I2V mode — resolved dims: height=%d, width=%d "
                    "(source: %s)",
                    height, width,
                    "user" if (_user_h is not None or _user_w is not None) else "first-frame",
                )
            else:
                # T2V mode: sensible default.
                height = _user_h if _user_h is not None else 544
                width  = _user_w if _user_w is not None else 960
                logger.info(
                    "[WeeLLM] T2V mode — resolved dims: height=%d, width=%d",
                    height, width,
                )

        # Write resolved dims back so both the cache key and diffusers see them.
        kwargs["height"] = height
        kwargs["width"]  = width

        if no_cache:
            return super().__call__(**kwargs)

        _video_cache = self._setup_cache(
            prompt, height, width, num_frames, steps, seed, save_every=cache_every, lora_weights=lora_weights
        )
        if fresh:
            _video_cache.clear_run()
        
        # Intercept encode_prompt
        _underlying = getattr(self, "_pipeline", self)
        _ep_hit = _video_cache.wrap_pipeline_encode_prompt(_underlying, device=str(self.device))
        # Fallback for ModularPipeline which has no encode_prompt but uses blocks
        if not _ep_hit and hasattr(_underlying, "_blocks") and hasattr(_underlying._blocks, "sub_blocks"):
            # Check if cache hit BEFORE running blocks
            cached = _video_cache.load_embeds()
            if cached is not None:
                _ep_hit = True
                for block_name, block in _underlying._blocks.sub_blocks.items():
                    if "TextEncoderStep" in block.__class__.__name__:
                        class CacheHitWrapper:
                            def __init__(self, block, cached_data):
                                self.block = block
                                self.cached_data = cached_data
                            def __getattr__(self, attr):
                                return getattr(self.block, attr)
                            def __call__(self, pipe, state):
                                logger.info("[VideoCache] Skipping ModularPipeline TextEncoder block (Cache HIT)")
                                block_state = self.block.get_block_state(state) if hasattr(self.block, "get_block_state") else getattr(state, getattr(self.block, "model_name", "unknown"), None)
                                if block_state is not None:
                                    device = getattr(pipe, "_execution_device", getattr(pipe, "device", "cuda"))
                                    embed_tensor = self.cached_data["embeds"].to(device)
                                    tag_tensor = self.cached_data.get("tags", torch.full((embed_tensor.shape[1],), getattr(pipe, "text_tag", 0), dtype=torch.long)).cpu()
                                    block_state.prompt_embeds = embed_tensor
                                    block_state.text_token_tags = tag_tensor
                                    if hasattr(state, "set"):
                                        state.set("prompt_embeds", embed_tensor)
                                        state.set("text_token_tags", tag_tensor)
                                return pipe, state
                        _underlying._blocks.sub_blocks[block_name] = CacheHitWrapper(block, cached)
                        break
            else:
                for block_name, block in _underlying._blocks.sub_blocks.items():
                    if "TextEncoderStep" in block.__class__.__name__:
                        class CacheMissWrapper:
                            def __init__(self, block, cache):
                                self.block = block
                                self.cache = cache
                            def __getattr__(self, attr):
                                return getattr(self.block, attr)
                            def __call__(self, pipe, state):
                                pipe, state = self.block(pipe, state)
                                embeds = None
                                tags = None
                                if hasattr(state, "values") and isinstance(state.values, dict):
                                    embeds = state.values.get("prompt_embeds")
                                    tags = state.values.get("text_token_tags")
                                    
                                if embeds is not None:
                                    self.cache.save_embeds({
                                        "embeds": embeds,
                                        "tags": tags
                                    })
                                return pipe, state
                        _underlying._blocks.sub_blocks[block_name] = CacheMissWrapper(block, _video_cache)
                        break

        # ModularPipeline ignores callback_on_step_end, so we recursively hook loop_step on any Loop block
        if hasattr(_underlying, "_blocks") and hasattr(_underlying._blocks, "sub_blocks"):
            def _patch_denoise_loops(blocks_dict):
                for block in blocks_dict.values():
                    if hasattr(block, "loop_step") and not getattr(block, "_weellm_patched_loop", False):
                        original_loop = getattr(block, "loop_step")
                        def _hooked_loop(self_block, pipe, state, i, t, orig=original_loop):
                            import os
                            resume_idx = int(os.environ.get("WEELLM_RESUME_STEP", "0"))
                            if i < resume_idx:
                                print(f"\n!!! Fast-Forwarding: Skipping Step {i} !!!")
                                if i == resume_idx - 1:
                                    resume_path = os.environ.get("WEELLM_RESUME_PATH", "")
                                    if resume_path and os.path.exists(resume_path):
                                        print(f"!!! Injecting Latents from {resume_path} !!!\n")
                                        import torch
                                        cached = torch.load(resume_path, map_location="cpu")
                                        if isinstance(cached, torch.Tensor):
                                            state.latents = cached.to(state.latents.device)
                                            state.noise_pred = torch.zeros_like(state.latents)
                                        elif isinstance(cached, dict) and "latents" in cached:
                                            state.latents = cached["latents"].to(state.latents.device)
                                            state.noise_pred = torch.zeros_like(state.latents)
                                            if "audio_latents" in cached and hasattr(state, "audio_latents"):
                                                state.audio_latents = cached["audio_latents"].to(getattr(state, "audio_latents").device)
                                        if hasattr(state, "audio_latents"):
                                            state.audio_noise_pred = torch.zeros_like(state.audio_latents)
                                return pipe, state
                            
                            pipe, state = orig(pipe, state, i=i, t=t)
                            callback = _video_cache.get_step_callback(resume_step=resume_step, resume_latents=resume_latents)
                            if callback is not None:
                                cb_kwargs = {"latents": getattr(state, "latents", None)}
                                if hasattr(state, "audio_latents"):
                                    cb_kwargs["audio_latents"] = getattr(state, "audio_latents", None)
                                
                                ret_cb = callback(pipe, i, t, cb_kwargs)
                                if ret_cb is not None:
                                    cb_kwargs = ret_cb
                                
                                if "latents" in cb_kwargs and cb_kwargs["latents"] is not None:
                                    state.latents = cb_kwargs["latents"]
                                if "audio_latents" in cb_kwargs and cb_kwargs["audio_latents"] is not None and hasattr(state, "audio_latents"):
                                    state.audio_latents = cb_kwargs["audio_latents"]
                            return pipe, state
                        block.loop_step = _hooked_loop.__get__(block, block.__class__)
                        block._weellm_patched_loop = True
                    elif hasattr(block, "sub_blocks"):
                        _patch_denoise_loops(block.sub_blocks)
            _patch_denoise_loops(_underlying._blocks.sub_blocks)
        
        if _ep_hit:
            logger.info("[VideoCache] Text encoder output restored from cache — skipped entirely.")
            try:
                from weellm.io.memory import evict_module as _evict_te
                _te_names = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
                for _te_name in _te_names:
                    _te_mod = getattr(_underlying, _te_name, None)
                    if _te_mod is not None and isinstance(_te_mod, torch.nn.Module):
                        _evict_te(_te_mod)
                        setattr(_underlying, _te_name, None)
                for _tok_name in ("tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"):
                    if hasattr(_underlying, _tok_name):
                        setattr(_underlying, _tok_name, None)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"[VideoCache] TE eviction failed: {e}")

        # Ensure output_type="latent" if supported
        _pipe_call = getattr(_underlying, "__call__", None) or _underlying.__class__.__call__
        if "output_type" in inspect.signature(_pipe_call).parameters:
            kwargs.setdefault("output_type", "latent")
            
        if lora_weights is not None:
            logger.info(f"Loading LoRA weights from {lora_weights}")
            
            from weellm.models.loras.lora_streamer import GenericLazyLoRALoader
            lazy_loader = GenericLazyLoRALoader(lora_weights)
            
            # 1. Apply to Transformer
            _tr_model = getattr(_underlying, "transformer", None) or getattr(_underlying, "unet", None)
            if _tr_model is not None:
                # The shard name for the whole transformer can just be "transformer"
                lazy_loader.apply_to_module(_tr_model, "transformer")
                
            # 2. Apply to Connectors if they exist
            _conn_model = getattr(_underlying, "connectors", None)
            if _conn_model is not None:
                lazy_loader.apply_to_module(_conn_model, "connectors")

        if _video_cache.has_final():
            logger.info("[VideoCache] Final-latents cache HIT — running decode-only.")
            _cached_final = _video_cache.load_final()
            _cached_latents = _cached_final.get("latents")
            if _cached_latents is not None and hasattr(_underlying, "vae") and _underlying.vae is not None:
                try:
                    _vae = _underlying.vae
                    _vproc = getattr(_underlying, "video_processor", None)
                    
                    from weellm.io.memory import evict_module as _evict
                    for _tr_name in ("transformer", "unet", "connectors"):
                        _tr = getattr(_underlying, _tr_name, None)
                        if _tr is not None: _evict(_tr)
                        
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    _dev = torch.device(str(self.device))
                    _lat = _cached_latents.to(device=_dev, dtype=_vae.dtype)
                    
                    if hasattr(_vae, "config"):
                        _lat = self._preprocess_latents_for_decode(_lat, _vae, kwargs)
                            
                            
                    with torch.no_grad():
                        _decoded = _vae.decode(_lat, return_dict=False)[0]
                        _video = _vproc.postprocess_video(_decoded, output_type="pil") if _vproc else _decoded
                        
                    _audio = None
                    if "audio_latents" in _cached_final and getattr(_underlying, "audio_vae", None):
                        try:
                            _audio_vae = _underlying.audio_vae
                            _vocoder = getattr(_underlying, "vocoder", None)
                            with torch.no_grad():
                                _audio_lat = _cached_final["audio_latents"].to(_audio_vae.dtype).to(_audio_vae.device)
                                if hasattr(_audio_vae, "config") and hasattr(_audio_vae.config, "latents_mean"):
                                    _a_mean = torch.tensor(_audio_vae.config.latents_mean, device=_audio_vae.device, dtype=_audio_vae.dtype).view(1, -1, 1)
                                    _a_std = torch.tensor(_audio_vae.config.latents_std, device=_audio_vae.device, dtype=_audio_vae.dtype).view(1, -1, 1)
                                    _audio_lat = _audio_lat * _a_std + _a_mean
                                _mel = _audio_vae.decode(_audio_lat, return_dict=False)[0]
                                if _vocoder is not None:
                                    _mel = _mel.to(_vocoder.dtype).to(_vocoder.device)
                                    _audio_out = _vocoder(_mel)
                                    if isinstance(_audio_out, tuple): _audio_out = _audio_out[0]
                                else:
                                    _audio_out = _mel.float().permute(1, 0, 2)
                                _audio = _audio_out.cpu().float()
                        except Exception as e:
                            logger.warning(f"[VideoCache] Audio decode failed: {e}")

                    _audio_rate = 48000
                    if hasattr(_underlying, "vocoder") and hasattr(_underlying.vocoder, "config"):
                        _audio_rate = getattr(_underlying.vocoder.config, "output_sampling_rate", 48000)

                    return WeeVideoResult(
                        frames=_video,
                        audio=_audio,
                        audio_rate=_audio_rate,
                        default_fps=kwargs.get("fps", kwargs.get("frame_rate", 24))
                    )
                except Exception as e:
                    logger.warning(f"[VideoCache] Decode-only failed: {e}")
                    kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
                    return super().__call__(**kwargs)
            else:
                kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
                return super().__call__(**kwargs)

        # Check for intermediate step cache
        resume_step = -1
        resume_latents = None
        if not fresh and not _video_cache.has_final():
            latest_file = _video_cache.latest_step_file()
            if latest_file is not None:
                resume_step = _video_cache.step_from_file(latest_file)
                resume_latents = _video_cache.load_step(latest_file)
                logger.info(f"[VideoCache] Resuming from step {resume_step}!")

                class SkipDenoisingWrapper:
                    def __init__(self, original, r_step):
                        self.original = original
                        self.resume_step = r_step
                        self.current_step = 0
                    def __getattr__(self, name):
                        return getattr(self.original, name)
                    def __call__(self, *args, **kw):
                        if self.current_step <= self.resume_step:
                            logger.info(f"[VideoCache] Fast-forwarding step {self.current_step}")
                            self.current_step += 1
                            latents = kw.get("hidden_states", args[0] if len(args) > 0 else None)
                            if latents is None: latents = kw.get("sample", None)
                            dummy = torch.zeros_like(latents) if latents is not None else None
                            
                            ret_dict = kw.get("return_dict", True)
                            if "audio_hidden_states" in kw and kw["audio_hidden_states"] is not None:
                                audio_dummy = torch.zeros_like(kw["audio_hidden_states"])
                                if not ret_dict: return (dummy, audio_dummy)
                                class DummyOut:
                                    sample = dummy
                                    audio_sample = audio_dummy
                                    def __getitem__(self, idx): return (self.sample, self.audio_sample)[idx]
                                return DummyOut()
                                
                            if not ret_dict: return (dummy,)
                            class DummyOut:
                                sample = dummy
                                def __getitem__(self, idx): return (self.sample,)[idx]
                            return DummyOut()
                            
                        self.current_step += 1
                        return self.original(*args, **kw)
                
                if hasattr(_underlying, "transformer"):
                    _underlying.transformer = SkipDenoisingWrapper(_underlying.transformer, resume_step)
                elif hasattr(_underlying, "unet"):
                    _underlying.unet = SkipDenoisingWrapper(_underlying.unet, resume_step)

        if resume_step >= 0:
            kwargs["callback_on_step_end"] = _video_cache.get_step_callback(resume_step=resume_step, resume_latents=resume_latents)
        else:
            kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
            
        # ModularPipelines don't support callback_on_step_end natively
        if hasattr(_underlying, "_blocks"):
            kwargs.pop("callback_on_step_end", None)
            
        out = super().__call__(**kwargs)
        
        try:
            _raw_latents = None
            _raw_audio = None
            if hasattr(out, "frames") and isinstance(out.frames, torch.Tensor):
                _raw_latents = out.frames
            elif hasattr(out, "videos") and isinstance(out.videos, torch.Tensor):
                _raw_latents = out.videos
            if hasattr(out, "audio") and isinstance(out.audio, torch.Tensor):
                _raw_audio = out.audio
                
            if _raw_latents is not None:
                _save_dict = {"latents": _raw_latents.cpu()}
                if _raw_audio is not None:
                    _save_dict["audio_latents"] = _raw_audio.cpu()
                _video_cache.save_final(_save_dict)
        except Exception as e:
            logger.debug(f"[VideoCache] Could not save final cache: {e}")
            
        if hasattr(out, "frames") and isinstance(out.frames, torch.Tensor):
            _vae = getattr(_underlying, "vae", None)
            _vproc = getattr(_underlying, "video_processor", None)
            
            if _vae and _vproc:
                try:
                    _dev = torch.device(str(self.device))
                    _lat = out.frames.to(device=_dev, dtype=_vae.dtype)
                    
                    _lat = self._preprocess_latents_for_decode(_lat, _vae, kwargs)
                            
                    from weellm.io.memory import evict_module
                    for _comp in ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "connectors"]:
                        _c = getattr(_underlying, _comp, None)
                        if _c: evict_module(_c)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    if user_output_type == "latent":
                        out.frames = _lat
                        return out
                    
                    with torch.no_grad():
                        _dec = _vae.decode(_lat, return_dict=False)[0]
                        _vid = _vproc.postprocess_video(_dec, output_type="pil")
                        
                    out.frames = _vid
                except Exception as e:
                    logger.warning(f"[VideoCache] Manual decode failed: {e}")
                    
        # Wrap the diffusers output in our clean WeeVideoResult
        _frames = getattr(out, "frames", None)
        if _frames is None and hasattr(out, "videos"):
            _frames = out.videos
            
        _audio = getattr(out, "audio", None)
        
        # If we have audio latents and a vocoder, manually decode
        if _audio is not None and isinstance(_audio, torch.Tensor) and _audio.dim() > 2 and user_output_type != "latent":
            _audio_vae = getattr(_underlying, "audio_vae", None)
            _vocoder = getattr(_underlying, "vocoder", None)
            
            _is_waveform = False
            if _audio_vae and hasattr(_audio_vae, "config") and hasattr(_audio_vae.config, "latent_channels"):
                # If channel dim does not match latent_channels, it's likely already a waveform
                if _audio.shape[-2] != _audio_vae.config.latent_channels:
                    _is_waveform = True
            
            if not _is_waveform and _audio_vae:
                try:
                    logger.info("[WeeLLM] Manually decoding audio latents using audio_vae...")
                    with torch.no_grad():
                        _audio = _audio.to(_audio_vae.dtype).to(_audio_vae.device)
                        if hasattr(_audio_vae, "config") and hasattr(_audio_vae.config, "latents_mean"):
                            _a_mean = torch.tensor(_audio_vae.config.latents_mean, device=_audio_vae.device, dtype=_audio_vae.dtype).view(1, -1, 1)
                            _a_std = torch.tensor(_audio_vae.config.latents_std, device=_audio_vae.device, dtype=_audio_vae.dtype).view(1, -1, 1)
                            _audio = _audio * _a_std + _a_mean
                        _mel = _audio_vae.decode(_audio, return_dict=False)[0]
                        if _vocoder is not None:
                            _mel = _mel.to(_vocoder.dtype).to(_vocoder.device)
                            _audio_out = _vocoder(_mel)
                            if isinstance(_audio_out, tuple): _audio_out = _audio_out[0]
                        else:
                            _audio_out = _mel.float().permute(1, 0, 2)
                        _audio = _audio_out.cpu().float()
                except Exception as e:
                    logger.warning(f"Audio manual decoding failed: {e}")
            elif not _audio_vae and not _is_waveform:
                logger.warning("Pipeline output audio latents, but audio_vae not found for manual decode.")
        
        # Unbatch if necessary
        if _frames is not None and isinstance(_frames, torch.Tensor) and _frames.dim() == 5:
            _frames = _frames[0]
        elif _frames is not None and isinstance(_frames, list) and len(_frames) > 0 and isinstance(_frames[0], list):
            _frames = _frames[0]
            
        _fps = kwargs.get("fps", kwargs.get("frame_rate", 24))
        
        _audio_rate = 48000
        if hasattr(_underlying, "vocoder") and hasattr(_underlying.vocoder, "config"):
            _audio_rate = getattr(_underlying.vocoder.config, "output_sampling_rate", 48000)

        # Return the wrapper if we successfully intercepted frames, otherwise return raw output
        if _frames is not None:
            return WeeVideoResult(
                frames=_frames,
                audio=_audio,
                default_fps=_fps,
                audio_rate=_audio_rate
            )
            
        return out
