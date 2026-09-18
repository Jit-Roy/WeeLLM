"""
weellm.pipelines.video.ltx2
===========================
Custom patched adapter for LTX-2.5 Video Pipeline.
Handles 8k+1 frame math and dummy audio injection.
"""

import math
import logging
import torch
from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeLTX2Pipeline(WeeVideoPipeline):
    
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from pathlib import Path
        import json
        import torch
        from diffusers import LTX2Pipeline
        from weellm.pipelines.weebasepipeline import WeeBasePipeline

        # Force skip downloading/loading the unneeded massive components by injecting a dummy _path.
        # This tells WeeBasePipeline that we are overriding them, so it removes them from HF downloads.
        kwargs.setdefault("prompt_enhancer_path", "DUMMY_SKIP_DOWNLOAD")
        
        # If model_dir is a Hugging Face Repo ID, resolve it to a local path now
        # so that our manual vocoder/connectors loading can find the local files.
        from weellm.io.utils import resolve_model_path
        skip_components = {k[:-5] for k, v in kwargs.items() if k.endswith("_path") and v is not None}
        model_dir = str(resolve_model_path(str(model_dir), skip_components=skip_components or None))
        
        model_dir_path = Path(model_dir)
        device = kwargs.get("device", "cuda")
        dtype  = kwargs.get("torch_dtype", torch.bfloat16)
        
        # 1. Inject Dummy Components for LTX specific missing parts to satisfy diffusers __init__
        class DummyComponent:
            def __call__(self, *args, **kwargs): return None
            def predict_num_frames(self, *args, **kwargs): return 121
            def to(self, *args, **kwargs): return self
        kwargs.setdefault("duration_head", DummyComponent())
        kwargs.setdefault("prompt_enhancer", DummyComponent())
        
        # 2. Load extra LTX components if they exist
        index_path = model_dir_path / "model_index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            if "vocoder" in index and "vocoder" not in kwargs:
                try:
                    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
                    vocoder = LTX2VocoderWithBWE.from_pretrained(
                        model_dir_path / "vocoder", torch_dtype=dtype
                    ).to(device)
                    kwargs["vocoder"] = vocoder
                except Exception as e:
                    logger.error("Failed to load vocoder: %s", e)
                    raise e
            
            if "connectors" in index and "connectors" not in kwargs:
                try:
                    from weellm.models.transformers.ltx2_connectors import LTX2ConnectorsStreamer
                    cache_to_ram = kwargs.get("cache_to_ram", False)
                    
                    conn = LTX2ConnectorsStreamer.from_pretrained(
                        model_dir_path / "connectors", device=device, dtype=dtype, cache_to_ram=cache_to_ram
                    )
                    conn_model = getattr(conn, "model", getattr(conn, "_model", conn))
                    conn_model = WeeBasePipeline._patch_to(conn_model)
                    conn_model._weellm_streamer = conn
                    kwargs["connectors"] = conn_model
                except Exception as e:
                    logger.warning("Failed to load connectors in WeeLTX2Pipeline: %s", e)
                    
            if "audio_vae" in index and "audio_vae" not in kwargs:
                try:
                    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
                    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                        model_dir_path / "audio_vae", torch_dtype=dtype
                    ).to(device)
                    kwargs["audio_vae"] = audio_vae
                except Exception as e:
                    logger.warning("Failed to load audio_vae: %s", e)
                    
        _temporal = kwargs.pop("temporal_upscaler", None)
        if "temporal_upscaler" in index and _temporal is None:
            try:
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _temporal = LTX2LatentUpsamplerModel.from_pretrained(
                    model_dir_path / "temporal_upscaler", torch_dtype=dtype
                ).to(device)
            except Exception as e:
                logger.warning("Failed to load temporal_upscaler: %s", e)

        pipe = super().from_pretrained(model_dir, **kwargs)
        
        if _temporal is not None:
            if isinstance(_temporal, str):
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _temporal = LTX2LatentUpsamplerModel.from_pretrained(_temporal, torch_dtype=dtype).to(device)
            # Attach to the underlying diffusers pipeline so kwargs.get('temporal_upscaler') is not needed
            setattr(pipe._pipeline, "temporal_upscaler", _temporal)
        
        # LTX-2.5 VAE decoding configuration:
        # The base WeeLLM class enables framewise decoding with chunk=9, stride=8 (1 frame overlap)
        # which causes visible temporal seams/jitter. But fully disabling it causes 5.81 GiB OOM
        # on 4 GB VRAM when latent upsamplers are used.
        # 
        # The real fix: use the VAE's built-in _temporal_tiled_decode which has proper blend_t()
        # linear crossfading at chunk boundaries. With large enough overlap the blending is invisible.
        # We use tile=33 sample frames, stride=17 (~50% overlap = 16-frame crossfade zone).
        vae = getattr(pipe._pipeline, "vae", None)
        if vae is not None and hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = True
            # tile_sample_min_num_frames must be multiple of temporal_compression_ratio (8) + 1
            # 33 sample frames = 4 latent frames chunk, 17 sample stride = 2 latent stride
            # Overlap = 33 - 17 = 16 sample frames of crossfade → invisible seam
            vae.tile_sample_min_num_frames = 33
            vae.tile_sample_stride_num_frames = 17
            # Disable spatial tiling (causes block seams) — let temporal tiling handle VRAM
            for _attr in ("tile_sample_min_width", "tile_sample_min_height", "tile_sample_min_size"):
                if hasattr(vae, _attr):
                    setattr(vae, _attr, 10_000)
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            logger.info(
                "      -> [WeeLLM/LTX2] VAE temporal tiling enabled: "
                "tile=33 sample frames, stride=17 (~50%% overlap). "
                "blend_t() crossfading active — no seam artifacts."
            )
        if hasattr(pipe._pipeline, "enable_vae_tiling"):
            pipe._pipeline.enable_vae_tiling()
            
        # Gemma 3/4 uses left-padding for prompts. Because it is a causal model, 
        # the padded tokens at the beginning of the sequence have nothing to attend to 
        # (their entire attention row is masked out). When PyTorch calculates Softmax 
        # on an entirely masked row (all -inf), it outputs NaN.
        # These NaNs ONLY exist on the padded tokens. The real tokens are perfectly healthy.
        # The downstream LTX transformer ignores these padded tokens anyway, but the NaNs 
        # trip up the WeeLLM cache corruption check. We simply zero them out.
        _orig_encode = pipe.encode_prompt
        def _safe_encode(*args, **kwargs):
            out = _orig_encode(*args, **kwargs)
            # out is (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)
            clean_out = []
            _target_dtype = getattr(pipe._pipeline, "dtype", dtype)
            for item in out:
                if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
                    clean_item = torch.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0)
                    clean_out.append(clean_item.to(_target_dtype))
                else:
                    clean_out.append(item)
            return tuple(clean_out)
                    
        pipe.encode_prompt = _safe_encode
        
        # 4. Automatically handle Data Type casting for Connectors (Handles Cache Hits)
        if hasattr(pipe._pipeline, "connectors"):
            _orig_conn_forward = pipe._pipeline.connectors.forward
            def _safe_conn_forward(*args, **kwargs):
                _target_dtype = getattr(pipe._pipeline, "dtype", dtype)
                new_args = tuple(a.to(_target_dtype) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args)
                new_kwargs = {k: (v.to(_target_dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
                return _orig_conn_forward(*new_args, **new_kwargs)
            pipe._pipeline.connectors.forward = _safe_conn_forward

        return pipe

    def __call__(self, prompt: str, **kwargs):
        # 0. Handle Pipeline Routing for Image/Video modalities
        _first_frame = kwargs.pop("first_frame", None)
        _last_frame = kwargs.pop("last_frame", None)
        # For backwards compatibility with standard Diffusers 'image'
        _image_fallback = kwargs.pop("image", None)
        if _first_frame is None and _image_fallback is not None:
            _first_frame = _image_fallback
        
        # Remove video kwarg if it somehow gets passed to prevent crashes
        kwargs.pop("video", None)

        if _first_frame is not None:
            _current_class_name = self._pipeline.__class__.__name__
            _model_dir = getattr(self._pipeline, "model_dir", None)
            _safe_encode = getattr(self._pipeline, "encode_prompt", None)
            
            if _last_frame is not None:
                # Both frames provided: Route to InContext Pipeline (Interpolation)
                if "InContext" not in _current_class_name:
                    from diffusers import LTX2InContextPipeline
                    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
                    import inspect
                    logger.info("Routing to LTX2InContextPipeline (Video Interpolation)...")
                    valid_keys = set(inspect.signature(LTX2InContextPipeline.__init__).parameters.keys())
                    safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                    self._pipeline = LTX2InContextPipeline(**safe_components)
                    if _model_dir:
                        self._pipeline.model_dir = _model_dir
                
                # Build the condition objects for the underlying pipeline
                from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
                kwargs["conditions"] = [
                    LTX2VideoCondition(frames=_first_frame, index=0, strength=1.0),
                    LTX2VideoCondition(frames=_last_frame, index=-1, strength=1.0)
                ]
                
            else:
                # Only first frame provided: Route to ImageToVideo Pipeline
                if "ImageToVideo" not in _current_class_name:
                    from diffusers import LTX2ImageToVideoPipeline
                    import inspect
                    logger.info("Routing to LTX2ImageToVideoPipeline (Image-to-Video)...")
                    valid_keys = set(inspect.signature(LTX2ImageToVideoPipeline.__init__).parameters.keys())
                    safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                    self._pipeline = LTX2ImageToVideoPipeline(**safe_components)
                    if _model_dir:
                        self._pipeline.model_dir = _model_dir
                
                kwargs["image"] = _first_frame
                
            if _safe_encode is not None:
                self._pipeline.encode_prompt = _safe_encode
        else:
            # Neither provided: Route back to TextToVideo if needed
            _current_class_name = self._pipeline.__class__.__name__
            if _current_class_name != "LTX2Pipeline":
                from diffusers import LTX2Pipeline
                import inspect
                logger.info("Routing back to LTX2Pipeline (Text-to-Video)...")
                _model_dir = getattr(self._pipeline, "model_dir", None)
                _safe_encode = getattr(self._pipeline, "encode_prompt", None)
                valid_keys = set(inspect.signature(LTX2Pipeline.__init__).parameters.keys())
                safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                self._pipeline = LTX2Pipeline(**safe_components)
                if _model_dir:
                    self._pipeline.model_dir = _model_dir
                if _safe_encode is not None:
                    self._pipeline.encode_prompt = _safe_encode

        # 1. Handle LTX-specific 8k+1 frame snapping
        _resolved_num_frames = kwargs.pop("num_frames", None)
        _duration = kwargs.pop("duration", None)
        
        if _resolved_num_frames is None and _duration is None:
            _duration = 5.0
            logger.info("  Auto-defaulting to %.1fs duration for LTX model", _duration)

        if _resolved_num_frames is None and _duration is not None:
            _fps = kwargs.get("fps", 24.0)
            _raw_frames = round(_duration * _fps)
            # LTX-2.5 requires frame count ≡ 1 (mod 8)
            _snapped = max(1, ((_raw_frames - 1 + 4) // 8) * 8 + 1)
            _resolved_num_frames = _snapped
            logger.info(
                "  Duration:  %.1fs @ %.0ffps → %d frames (snapped to 8k+1)",
                _duration, _fps, _resolved_num_frames,
            )
            
        kwargs["num_frames"] = _resolved_num_frames
        
        # 2. Inject Dummy Components for LTX specific missing parts
        _CALLABLE_COMPONENT_NAMES = ("duration_head", "prompt_enhancer")
        for _comp_name in _CALLABLE_COMPONENT_NAMES:
            if getattr(self._pipeline, _comp_name, None) is None:
                class DummyComponent:
                    def __call__(self, *args, **kwargs): return None
                    def predict_num_frames(self, *args, **kwargs): return 121
                setattr(self._pipeline, _comp_name, DummyComponent())
                logger.debug(
                    "[WeeLLM] Patched None '%s' with no-op dummy component (LTX adapter).",
                    _comp_name,
                )
                
        # 3. Delegate to the shared generic video generation loop in the base class
        return super().__call__(prompt=prompt, **kwargs)

    def _preprocess_latents_for_decode(self, latents, vae, kwargs):
        """
        LTX-2.5 specific latent preprocessing.
        LTX latents are pre-scaled, so we skip standard scaling/shifting.
        We also handle the optional latent upsampler and temporal upscaler here,
        and dynamically configure VAE temporal tiling to minimize wall-clock time.
        """
        temporal_upscaler = kwargs.get("temporal_upscaler", getattr(self._pipeline, "temporal_upscaler", None))
        if temporal_upscaler:
            logger.info("Upsampling latents temporally using %s", temporal_upscaler)
            
            _dev = latents.device
            _dtype = vae.dtype if vae else latents.dtype
            
            _ups_model = temporal_upscaler
            if isinstance(_ups_model, str):
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _ups_model = LTX2LatentUpsamplerModel.from_pretrained(_ups_model).to(device=_dev, dtype=_dtype)
            
            _ups_model = _ups_model.to(device=_dev, dtype=_dtype)
            with torch.no_grad():
                latents = _ups_model(latents)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        latent_upsampler = kwargs.get("latent_upsampler", None)
        if latent_upsampler:
            logger.info("Upsampling latents using %s", latent_upsampler)
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
            
            _dev = latents.device
            _dtype = vae.dtype if vae else latents.dtype
            
            _ups_model = LTX2LatentUpsamplerModel.from_pretrained(latent_upsampler).to(device=_dev, dtype=_dtype)
            with torch.no_grad():
                latents = _ups_model(latents)
            del _ups_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # VAE Block CPU-RAM Cache
        # Root cause of disk bottleneck:
        #   _block_post_hook calls _evict_keys -> set_module_tensor_to_device("meta")
        #   This evicts block weights entirely. Every tile is a cold disk read.
        #
        # The pipeline stores lazy_vae.MODEL (not the streamer) as pipeline.vae,
        # so `vae.seeker` doesn't exist. The streamer stays alive only via PyTorch's
        # hook reference system (hooks hold bound-method refs back to the streamer).
        #
        # Fix: Find the streamer through vae.decoder.mid_block._forward_pre_hooks,
        # then monkey-patch streamer.seeker.get_tensors with a CPU RAM cache.
        #   - First call (pre-warm): reads from disk, saves a CPU clone per key.
        #   - ALL subsequent tile calls: fast CPU->GPU H2D transfer only (~50ms).
        #
        # Only installed once per pipe session via a sentinel flag on the streamer.
        _vae_streamer = None
        _vae_decoder = getattr(vae, "decoder", None)
        if _vae_decoder is not None and getattr(_vae_decoder, "mid_block", None) is not None:
            for _hook_fn in _vae_decoder.mid_block._forward_pre_hooks.values():
                if callable(_hook_fn) and hasattr(_hook_fn, "__self__") and hasattr(_hook_fn.__self__, "seeker"):
                    _vae_streamer = _hook_fn.__self__
                    break

        if _vae_streamer is not None and not getattr(_vae_streamer, "_weellm_ram_cache_installed", False):
            try:
                _cpu_weight_cache: dict = {}
                _orig_get_tensors = _vae_streamer.seeker.get_tensors

                def _ram_cached_get_tensors(keys, device, dtype):
                    missing = [k for k in keys if k not in _cpu_weight_cache]
                    block_tag = keys[0].rsplit(".", 1)[0] if keys else "?"
                    if missing:
                        logger.info(
                            "[VAE Cache] MISS  %s — %d/%d keys absent. Reading from disk...",
                            block_tag, len(missing), len(keys)
                        )
                        # Cold read — load from disk, save pinned CPU clone for future tiles.
                        # pin_memory() = page-locked RAM the OS CANNOT swap to pagefile.
                        # Without pinning, tensors get paged out instantly under RAM pressure,
                        # turning H2D transfers into slow pagefile reads (88-117s per block!).
                        loaded = _orig_get_tensors(missing, device=device, dtype=dtype)
                        save_ok = 0
                        for k, v in loaded.items():
                            try:
                                cpu_v = v.detach().cpu()
                                try:
                                    cpu_v = cpu_v.pin_memory()  # page-lock: no swap possible
                                except Exception as _pe:
                                    logger.warning(
                                        "[VAE Cache] pin_memory FAILED for %s (numel=%d): %s",
                                        k, cpu_v.numel(), _pe
                                    )
                                _cpu_weight_cache[k] = cpu_v
                                save_ok += 1
                            except Exception as _ce:
                                logger.warning("[VAE Cache] Failed to save key %s to CPU cache: %s", k, _ce)
                        logger.info(
                            "[VAE Cache] Cached %d/%d loaded keys to CPU RAM (total cache: %d keys).",
                            save_ok, len(loaded), len(_cpu_weight_cache)
                        )
                    else:
                        vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                        logger.info(
                            "[VAE Cache] HIT   %s — all %d keys in CPU RAM (pinned=%s). VRAM used: %.1f MB",
                            block_tag, len(keys),
                            all(_cpu_weight_cache[k].is_pinned() for k in keys if k in _cpu_weight_cache),
                            vram_mb
                        )
                    # Serve all from cache. Pinned tensors: H2D via DMA, no pagefile reads.
                    result = {}
                    for k in keys:
                        if k in _cpu_weight_cache:
                            try:
                                t = _cpu_weight_cache[k]
                                result[k] = t.to(
                                    device=device,
                                    dtype=dtype if t.is_floating_point() else t.dtype,
                                    non_blocking=True,
                                )
                            except Exception as _se:
                                logger.warning("[VAE Cache] Failed to serve key %s from cache: %s", k, _se)
                    return result

                _vae_streamer.seeker.get_tensors = _ram_cached_get_tensors
                _vae_streamer._weellm_ram_cache_installed = True
                logger.info(
                    "[WeeLLM/LTX2] VAE block RAM cache installed (streamer found via hook ref). "
                    "Decoder blocks will be read from disk ONCE, then served from CPU RAM."
                )
            except Exception as _e:
                logger.warning("[WeeLLM/LTX2] RAM cache install failed (non-critical): %s", _e)

        # ── Pre-VAE VRAM Reclaim (FINAL FIX) ─────────────────────────────────────
        # DIAGNOSIS (from debug_te.py):
        #   pipeline.text_encoder = Gemma3TextModel (IS an nn.Module)
        #   embed_tokens.weight   = [262144, 3840] → 1920 MB  (on THIS module directly)
        #   orphaned gc tensor    = [4096, 188160] → 1470 MB  (failed embed_audio placement)
        #
        # ROOT CAUSE OF ALL PREVIOUS FAILURES:
        #   `getattr(_te, "model", ...)` → Gemma3TextModel.model → Gemma3Model (inner body)
        #   embed_tokens is on Gemma3TextModel, NOT on Gemma3TextModel.model!
        #   So we always iterated the WRONG module and found 0 CUDA params.
        #
        # FIX: iterate _te.parameters() DIRECTLY (no .model/./_model indirection).
        #   Also gc-scan for orphaned tensors from failed embed_audio/vision placements.
        import gc as _gc
        _vram_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        _seen_ptrs   = set()
        _total_freed = 0.0

        # Mark the latents ptr so we don't accidentally zero it in the gc scan below
        _latents_ptr = latents.data_ptr() if latents is not None else -1

        # 1. Zero parameters from each text encoder module directly
        for _te_key in ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"):
            _te = getattr(self._pipeline, _te_key, None)
            if _te is None or not isinstance(_te, torch.nn.Module):
                continue
            _freed = 0.0
            # Iterate _te directly — embed_tokens is on the TOP-LEVEL module, not .model!
            for _p in list(_te.parameters()):
                ptr = _p.data_ptr()
                if _p.is_cuda and ptr not in _seen_ptrs:
                    _seen_ptrs.add(ptr)
                    _freed += _p.element_size() * _p.nelement() / 1024**2
                    _p.data = torch.empty(0, dtype=_p.dtype)
            for _b in list(_te.buffers()):
                ptr = _b.data_ptr()
                if _b.is_cuda and ptr not in _seen_ptrs:
                    _seen_ptrs.add(ptr)
                    _freed += _b.element_size() * _b.nelement() / 1024**2
                    _b.data = torch.empty(0, dtype=_b.dtype)
            if _freed > 0:
                _total_freed += _freed
                logger.info("[WeeLLM/LTX2] Freed %.1f MB from %s.", _freed, _te_key)

        # 2. gc-scan for orphaned CUDA tensors (e.g. failed embed_audio/vision placements
        #    that were loaded to GPU but never registered in any nn.Module).
        #    SKIP: the latents tensor, anything already freed above.
        _orphan_freed = 0.0
        for _obj in _gc.get_objects():
            try:
                if not isinstance(_obj, torch.Tensor) or not _obj.is_cuda:
                    continue
                ptr = _obj.data_ptr()
                if ptr in _seen_ptrs or ptr == _latents_ptr:
                    continue
                mb = _obj.element_size() * _obj.nelement() / 1024**2
                if mb > 100:   # only large orphaned tensors
                    _seen_ptrs.add(ptr)
                    _orphan_freed += mb
                    _obj.data = torch.empty(0, dtype=_obj.dtype)
            except Exception:
                pass
        if _orphan_freed > 0:
            _total_freed += _orphan_freed
            logger.info("[WeeLLM/LTX2] Freed %.1f MB orphaned CUDA tensors (embed_audio/vision).", _orphan_freed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _vram_after = torch.cuda.memory_allocated() / 1024**2
            logger.info("[WeeLLM/LTX2] VRAM: %.1f → %.1f MB (freed %.1f MB before VAE).",
                        _vram_before, _vram_after, _total_freed)

        # ── Debug: save latents to disk once for VAE-only debugging ──────────
        import os as _os
        _latents_save_path = "debug_latents.pt"
        if not _os.path.exists(_latents_save_path):
            torch.save(latents.cpu(), _latents_save_path)
            logger.info("[WeeLLM/LTX2] Saved latents to %s for VAE debug. Shape: %s", _latents_save_path, tuple(latents.shape))

        if vae is not None and hasattr(vae, "_decode") and hasattr(vae, "use_framewise_decoding"):

            logger.info("[WeeLLM/LTX2] Pre-warming VAE decoder (populating CPU RAM cache)...")
            try:
                import gc
                _lat_ch = getattr(getattr(vae, "config", None), "latent_channels", 128)
                _dummy_z = torch.zeros(1, _lat_ch, 2, 2, 2, device=latents.device, dtype=latents.dtype)
                _was_fw = vae.use_framewise_decoding
                vae.use_framewise_decoding = False
                try:
                    with torch.no_grad():
                        vae._decode(_dummy_z, return_dict=False)
                except Exception as _e:
                    logger.debug("[WeeLLM/LTX2] Pre-warm inner error (harmless): %s", _e)
                finally:
                    vae.use_framewise_decoding = _was_fw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                _cache_ref = locals().get("_cpu_weight_cache")
                logger.info(
                    "[WeeLLM/LTX2] Pre-warm complete — CPU RAM cache has %s keys.",
                    len(_cache_ref) if _cache_ref is not None else "N/A (cache not installed)"
                )
            except Exception as _e:
                logger.warning("[WeeLLM/LTX2] Pre-warm failed (non-critical, continuing): %s", _e)
        
        # ── Adaptive VAE Temporal Tiling ──────────────────────────────────────────
        # Cost model with pinned CPU RAM cache:
        #   H2D transfer ≈ 0.1–2s per block (PCIe DMA from pinned RAM, no CUDA stall)
        #   VRAM during tile decode = baseline + activations + block weights
        #   Key constraint: VRAM must stay below total_vram - OS_overhead - latent_size
        #
        # Per-tile decoded output (float32 RGB):
        # ── VAE Temporal Tiling: dynamic tile sizing ───────────────────────────────
        # Always use framewise (temporal) tiling — the tile size is computed from
        # actual free VRAM measured right now (post-TE-eviction), scaled to the
        # real video resolution. No fixed VRAM tiers; works for any video length
        # on any GPU.
        #
        # Budget model (streaming: only 2 tiles in VRAM at once):
        #   bytes_per_frame  = 3 × H × W × 4   (float32 RGB output of decoder)
        #   peak_vram        = 2 × tile_frames × bytes_per_frame
        #                      + activation_overhead (≈ 2× tile output, empirical)
        #   total            ≈ 4 × tile_frames × bytes_per_frame ≤ free_vram × safety
        #   → max_tile_frames = free_vram × safety / (4 × bytes_per_frame)
        #
        # tile_frames must satisfy: tile = temporal_compression_ratio × k + 1
        # Minimum overlap of 16 sample frames for smooth blend_t() crossfade.
        # Minimum tile = 33 (= 8×4+1) so stride ≥ 17 and overlap = 16.
        # ─────────────────────────────────────────────────────────────────────────
        if vae is not None and hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = True  # always tile; size computed below

            _tcr      = getattr(vae, "temporal_compression_ratio", 8)
            _scr      = getattr(vae, "spatial_compression_ratio", 32)
            _overlap  = 16   # sample frames — keeps blend_t() crossfade invisible
            _min_tile = _tcr * 2 + 1   # = 17 for tcr=8; stride would be 1 frame

            # Actual output resolution (may differ from 480×832 for other resolutions)
            _lat_h = latents.shape[3]
            _lat_w = latents.shape[4]
            _H = _lat_h * _scr
            _W = _lat_w * _scr
            _bytes_per_frame = 3 * _H * _W * 4  # float32 RGB

            # Free VRAM right now (after TE eviction + empty_cache above)
            if torch.cuda.is_available():
                _free_vram, _ = torch.cuda.mem_get_info()
            else:
                _free_vram = 0

            if _free_vram > 0 and _bytes_per_frame > 0:
                # Use 40% of free VRAM for tile output budget (leaves room for
                # block weights + intermediate activations loaded by the streamer)
                _max_tile = int(_free_vram * 0.40 / _bytes_per_frame)
                # Snap down to valid: tile = _tcr × k + 1
                _k = max(1, (_max_tile - 1) // _tcr)
                _tile = _tcr * _k + 1
                # Clamp: minimum _min_tile, no hard upper cap (let VRAM decide)
                _tile = max(_min_tile, _tile)
            else:
                _tile = 33  # safe fallback if no CUDA

            _stride = max(1, _tile - _overlap)
            vae.tile_sample_min_num_frames    = _tile
            vae.tile_sample_stride_num_frames = _stride

            _tile_mb   = _tile * _bytes_per_frame / 1024**2
            _n_tiles   = max(1, (latents.shape[2] - (_tile // _tcr)) // (_stride // _tcr) + 1)
            logger.info(
                "[WeeLLM/LTX2] VAE temporal tiling: tile=%d frames, stride=%d, overlap=%d "
                "(~%d tiles, ~%.0f MB/tile, free VRAM %.0f MB).",
                _tile, _stride, _overlap, _n_tiles, _tile_mb, _free_vram / 1024**2,
            )

        # ── Streaming Temporal Tiled Decode patch ─────────────────────────────────
        # Replaces diffusers' _temporal_tiled_decode() which accumulates ALL decoded
        # tiles in row[] on GPU before blending. Instead, blend on-the-fly and
        # offload completed stride slices to CPU RAM immediately.
        # Peak VRAM overhead = 2 tiles (current + previous), not N tiles.
        if vae is not None and hasattr(vae, "_temporal_tiled_decode"):
            _vae_ref = vae

            def _streaming_temporal_tiled_decode(z, temb, causal=None, return_dict=True):
                from diffusers.models.autoencoders.autoencoder_kl_ltx2 import DecoderOutput
                batch_size, num_channels, num_frames, height, width = z.shape
                num_sample_frames = (num_frames - 1) * _vae_ref.temporal_compression_ratio + 1

                tile_latent_min_num_frames    = _vae_ref.tile_sample_min_num_frames    // _vae_ref.temporal_compression_ratio
                tile_latent_stride_num_frames = _vae_ref.tile_sample_stride_num_frames // _vae_ref.temporal_compression_ratio
                tile_latent_min_height = _vae_ref.tile_sample_min_height // _vae_ref.spatial_compression_ratio
                tile_latent_min_width  = _vae_ref.tile_sample_min_width  // _vae_ref.spatial_compression_ratio
                blend_num_frames = _vae_ref.tile_sample_min_num_frames - _vae_ref.tile_sample_stride_num_frames

                result_chunks = []   # completed stride-sized slices (on CPU)
                prev_decoded  = None # previous tile's decoded tensor (on GPU)
                tile_idx      = 0

                for i in range(0, num_frames, tile_latent_stride_num_frames):
                    tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]

                    if _vae_ref.use_tiling and (tile.shape[-1] > tile_latent_min_width
                                               or tile.shape[-2] > tile_latent_min_height):
                        decoded = _vae_ref.tiled_decode(tile, temb, causal=causal, return_dict=True).sample
                    else:
                        decoded = _vae_ref.decoder(tile, temb, causal=causal)

                    if i > 0:
                        decoded = decoded[:, :, :-1, :, :]

                    if prev_decoded is not None:
                        blended = _vae_ref.blend_t(prev_decoded, decoded, blend_num_frames)
                        chunk   = blended[:, :, : _vae_ref.tile_sample_stride_num_frames, :, :]
                        result_chunks.append(chunk.cpu())
                        del prev_decoded, blended, chunk
                        torch.cuda.empty_cache()
                        logger.debug(
                            "[WeeLLM/LTX2][StreamVAE] Tile %d blended+offloaded. VRAM: %.0f MB",
                            tile_idx, torch.cuda.memory_allocated() / 1024**2,
                        )
                    else:
                        chunk = decoded[:, :, : _vae_ref.tile_sample_stride_num_frames + 1, :, :]
                        result_chunks.append(chunk.cpu())
                        del chunk
                        torch.cuda.empty_cache()

                    prev_decoded = decoded
                    tile_idx    += 1

                # Last tile: no next tile to blend with, keep as-is
                if prev_decoded is not None:
                    result_chunks.append(prev_decoded.cpu())
                    del prev_decoded
                    torch.cuda.empty_cache()

                logger.info(
                    "[WeeLLM/LTX2][StreamVAE] All %d tiles decoded. Assembling on GPU...",
                    len(result_chunks),
                )
                dec = torch.cat([c.to(z.device) for c in result_chunks], dim=2)
                del result_chunks
                dec = dec[:, :, :num_sample_frames]

                if not return_dict:
                    return (dec,)
                return DecoderOutput(sample=dec)

            vae._temporal_tiled_decode = _streaming_temporal_tiled_decode

        return latents


    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
        """Intercept diffusers LoRA loading to preserve WeeLLM LiveSeeker hooks."""
        from weellm.models.loras.lora_streamer import GenericLazyLoRALoader
        logger.info(f"[WeeLLM] Intercepted load_lora_weights for {pretrained_model_name_or_path_or_dict}")
        
        lazy_loader = GenericLazyLoRALoader(pretrained_model_name_or_path_or_dict)
        _tr_model = getattr(self._pipeline, "transformer", None)
        if _tr_model is not None:
            lazy_loader.apply_to_module(_tr_model, "transformer")
            
        _conn_model = getattr(self._pipeline, "connectors", None)
        if _conn_model is not None:
            lazy_loader.apply_to_module(_conn_model, "connectors")
