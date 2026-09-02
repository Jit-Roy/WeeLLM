"""
weellm.pipelines.video.ltx2
===========================
Custom patched adapter for LTX-2.5 Video Pipeline.
Handles 8k+1 frame math and dummy audio injection.
"""

import math
import logging
import torch
from weellm.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeLTX2Pipeline(WeeVideoPipeline):
    
    def __call__(self, prompt: str, **kwargs):
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
        
        # 2. Inject Dummy Components for LTX specific missing audio parts
        _CALLABLE_COMPONENT_NAMES = ("vocoder", "duration_head", "prompt_enhancer")
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
