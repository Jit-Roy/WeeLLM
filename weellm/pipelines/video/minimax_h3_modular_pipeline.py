"""
weellm.pipelines.video.minimax_h3_modular_pipeline
==================================================
Custom patched adapter for MiniMax-H3 Modular Pipeline.
Uses WeeBasePipeline's automatic interception to route standard
diffusers MiniMax components to WeeLLM layer-by-layer streamers.
"""

import logging
from weellm.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeMiniMaxPipeline(WeeVideoPipeline):
    
    def __call__(self, prompt: str, **kwargs):
        # We rely on PyTorch's native SDPA (Flash/MemEfficient) to handle VRAM implicitly,
        # rather than using slow Python loops.
        # 1. Add empty dummy components for audio if missing
        # MiniMax ModularPipeline relies on `audio_vae` for t2va and fl2va,
        # but if we are doing video-only, we might need to bypass it.
        _CALLABLE_COMPONENT_NAMES = ("audio_vae", "audio_scheduler")
        for _comp_name in _CALLABLE_COMPONENT_NAMES:
            if getattr(self._pipeline, _comp_name, None) is None:
                if _comp_name == "audio_scheduler":
                    # Just reuse the video scheduler for dummy audio scheduling
                    setattr(self._pipeline, _comp_name, self._pipeline.scheduler)
                else:
                    class DummyComponent:
                        config = type('DummyConfig', (), {'latent_channels': 32, 'sampling_rate': 32000})()
                        def __call__(self, *args, **kwargs): return None
                        def encode(self, *args, **kwargs):
                            import torch
                            class Output:
                                latent_dist = type('Dist', (), {'sample': lambda: torch.zeros(1, 32, 1).to(self._pipeline.device)})()
                            return Output()
                    setattr(self._pipeline, _comp_name, DummyComponent())
                logger.debug(
                    "[WeeLLM] Patched None '%s' with dummy component.",
                    _comp_name,
                )

        # 2. MiniMax doesn't support guidance_scale since it's guidance-distilled
        kwargs.pop("guidance_scale", None)
        kwargs.pop("negative_prompt", None)
        kwargs.pop("image_guidance_scale", None)
        kwargs.pop("_video_cache", None)
        kwargs.pop("callback_on_step_end", None)
        
        # 3. Delegate to the shared generic video generation loop in the base class
        return super().__call__(prompt=prompt, **kwargs)
