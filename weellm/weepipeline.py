"""
weepipeline.py -- WeePipeline for Text-to-Image generation.
"""

from typing import Optional
import torch
import logging

from weellm.pipeline import WeeBasePipeline

logger = logging.getLogger("weellm")

class WeePipeline(WeeBasePipeline):
    """
    Text-to-Image WeePipeline.
    """
    
    @classmethod
    def _get_diffusers_pipeline_class(cls, index: dict) -> str:
        pipeline_class_name = index.get("_class_name")
        if not pipeline_class_name:
            raise ValueError("No _class_name found in model_index.json")
        return pipeline_class_name

    def generate(self, prompt: str, **kwargs):
        """
        Convenience wrapper that calls the pipeline and returns the first image.

        Parameters
        ----------
        prompt:
            Text prompt for image generation.
        seed:
            Optional integer random seed (extracted from kwargs).
        **kwargs:
            Any additional arguments forwarded to the diffusers pipeline call.

        Returns
        -------
        ``PIL.Image.Image`` — the first generated image.
        """
        seed = kwargs.pop("seed", None)
        generator: Optional[torch.Generator] = None
        if seed is not None:
            device = getattr(self._pipeline, "device", torch.device("cpu"))
            generator = torch.Generator(device=device).manual_seed(seed)

        if self._pipeline.__class__.__name__ == "ErnieImagePipeline" and kwargs.get("use_pe", False):
            logger.warning("[WeeLLM] Prompt Enhancer (PE) is currently disabled for ErnieImagePipeline due to performance constraints.")
            kwargs["use_pe"] = False

        out = self._pipeline(prompt=prompt, generator=generator, **kwargs)
        if hasattr(out, "images"):
            return out.images[0]
        return out[0][0]
