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

from weellm.weebasepipeline import WeeBasePipeline

logger = logging.getLogger("weellm")

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
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                class_name = json.load(f).get("_class_name", "")
                
        VIDEO_MODELS = {
            "LTXVideoPipeline": {
                "log": "Text-to-Video (LTX-2.5)",
                "module": "weellm.pipelines.video.ltx_video_pipeline",
                "class": "WeeLTX2Pipeline",
            },
            "MiniMaxH3ModularPipeline": {
                "log": "Text-to-Video+Audio (MiniMax-H3)",
                "module": "weellm.pipelines.video.minimax_h3_modular_pipeline",
                "class": "WeeMiniMaxPipeline",
            },
            "WanPipeline": {
                "log": "Text-to-Video (Wan)",
                "module": "weellm.weevideopipeline",
                "class": "WeeVideoPipeline",
            },
            "CogVideoXPipeline": {
                "log": "Text-to-Video (CogVideoX)",
                "module": "weellm.weevideopipeline",
                "class": "WeeVideoPipeline",
            },
        }

        # If this method is called strictly on WeeVideoPipeline (the base), do the routing.
        # But if it's called on a subclass directly (e.g. WeeLTX2Pipeline.from_pretrained),
        # we skip the routing to avoid infinite loops and just construct it.
        if cls is WeeVideoPipeline and class_name in VIDEO_MODELS:
            info = VIDEO_MODELS[class_name]
            logger.info("  Mode:     %s", info["log"])
            
            module = importlib.import_module(info["module"])
            pipeline_class = getattr(module, info["class"])
            return pipeline_class.from_pretrained(model_dir, **kwargs)
            
        return super().from_pretrained(model_dir, **kwargs)

    def _setup_cache(self, prompt, height, width, num_frames, steps, seed, save_every=1, cache_root=None):
        from weellm.helpers.video_cache import VideoStepCache
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
        
        generator = kwargs.get("generator")
        seed = kwargs.pop("seed", generator.initial_seed() if generator is not None else 42)
        
        prompt = kwargs.get("prompt", "")
        height = kwargs.get("height", 544)
        width = kwargs.get("width", 960)
        num_frames = kwargs.get("num_frames", 121)
        steps = kwargs.get("num_inference_steps", 6)

        if no_cache:
            return super().__call__(**kwargs)

        _video_cache = self._setup_cache(
            prompt, height, width, num_frames, steps, seed, save_every=cache_every
        )
        if fresh:
            _video_cache.clear_run()
        kwargs["_video_cache"] = _video_cache
        
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
                            callback = _video_cache.get_step_callback()
                            if callback is not None:
                                latents = getattr(state, "latents", None)
                                cb_kwargs = {"latents": latents}
                                callback(pipe, i, t, cb_kwargs)
                            return pipe, state
                        block.loop_step = _hooked_loop.__get__(block, block.__class__)
                        block._weellm_patched_loop = True
                    elif hasattr(block, "sub_blocks"):
                        _patch_denoise_loops(block.sub_blocks)
            _patch_denoise_loops(_underlying._blocks.sub_blocks)
        
        if _ep_hit:
            logger.info("[VideoCache] Text encoder output restored from cache — skipped entirely.")
            try:
                from weellm.memory import evict_module as _evict_te
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

        if _video_cache.has_final():
            logger.info("[VideoCache] Final-latents cache HIT — running decode-only.")
            _cached_final = _video_cache.load_final()
            _cached_latents = _cached_final.get("latents")
            if _cached_latents is not None and hasattr(_underlying, "vae") and _underlying.vae is not None:
                try:
                    _vae = _underlying.vae
                    _vproc = getattr(_underlying, "video_processor", None)
                    
                    from weellm.memory import evict_module as _evict
                    for _tr_name in ("transformer", "unet", "connectors"):
                        _tr = getattr(_underlying, _tr_name, None)
                        if _tr is not None: _evict(_tr)
                        
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    _dev = torch.device(str(self.device))
                    _lat = _cached_latents.to(device=_dev, dtype=_vae.dtype)
                    
                    if hasattr(_vae, "config"):
                        _scaling = getattr(_vae.config, "scaling_factor", 1.0)
                        _shift = getattr(_vae.config, "shift_factor", 0.0)
                        if _scaling != 1.0 or _shift != 0.0:
                            _lat = (_lat / _scaling) - _shift
                            
                    with torch.no_grad():
                        _decoded = _vae.decode(_lat, return_dict=False)[0]
                        _video = _vproc.postprocess_video(_decoded, output_type="pil") if _vproc else _decoded
                        
                    class _CachedOutput:
                        frames = [_video]
                        audio = None
                        sampling_rate = 24000
                    return _CachedOutput()
                except Exception as e:
                    logger.warning(f"[VideoCache] Decode-only failed: {e}")
                    kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
                    return super().__call__(**kwargs)
            else:
                kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
                return super().__call__(**kwargs)

        kwargs["callback_on_step_end"] = _video_cache.get_step_callback()
        
        # ModularPipelines don't support callback_on_step_end or _video_cache natively
        if hasattr(_underlying, "_blocks"):
            kwargs.pop("_video_cache", None)
            kwargs.pop("callback_on_step_end", None)
            
        out = super().__call__(**kwargs)
        
        try:
            _raw_latents = None
            if hasattr(out, "frames") and isinstance(out.frames, torch.Tensor):
                _raw_latents = out.frames
            elif hasattr(out, "videos") and isinstance(out.videos, torch.Tensor):
                _raw_latents = out.videos
            if _raw_latents is not None:
                _video_cache.save_final({"latents": _raw_latents.cpu()})
        except Exception as e:
            logger.debug(f"[VideoCache] Could not save final cache: {e}")
            
        if hasattr(out, "frames") and isinstance(out.frames, torch.Tensor):
            _vae = getattr(_underlying, "vae", None)
            _vproc = getattr(_underlying, "video_processor", None)
            
            if _vae and _vproc:
                try:
                    _dev = torch.device(str(self.device))
                    _lat = out.frames.to(device=_dev, dtype=_vae.dtype)
                    
                    if hasattr(_vae, "config"):
                        _scaling = getattr(_vae.config, "scaling_factor", 1.0)
                        _shift = getattr(_vae.config, "shift_factor", 0.0)
                        if _scaling != 1.0 or _shift != 0.0:
                            _lat = (_lat / _scaling) - _shift
                            
                    from weellm.memory import evict_module
                    for _comp in ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "connectors"]:
                        _c = getattr(_underlying, _comp, None)
                        if _c: evict_module(_c)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        _dec = _vae.decode(_lat, return_dict=False)[0]
                        _vid = _vproc.postprocess_video(_dec, output_type="pil")
                        
                    out.frames = _vid
                except Exception as e:
                    logger.warning(f"[VideoCache] Manual decode failed: {e}")
                    
        return out
