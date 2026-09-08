"""
weeimagetoimagepipeline.py -- WeeImageToImagePipeline for Image-to-Image and Image Editing.
"""

from typing import Optional
import torch
import logging
from PIL import Image

from weellm.weebasepipeline import WeeBasePipeline

logger = logging.getLogger("weellm")

IMG2IMG_MAPPING = {
    "StableDiffusionPipeline": "StableDiffusionImg2ImgPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusion3Pipeline": "StableDiffusion3Img2ImgPipeline",
    "LongCatImagePipeline": "LongCatImageEditPipeline",
    "HiDreamImagePipeline": "HiDreamImageEditingPipeline",
    "FluxPipeline": "FluxImg2ImgPipeline",
    "Flux2KleinPipeline": "Flux2KleinPipeline", 
    "FluxFillPipeline": "FluxFillPipeline",
}

class WeeImageToImagePipeline(WeeBasePipeline):
    """
    Image-to-Image WeeImageToImagePipeline.
    """
    
    @classmethod
    def _get_diffusers_pipeline_class(cls, index: dict) -> str:
        base_class_name = index.get("_class_name")
        if not base_class_name:
            raise ValueError("No _class_name found in model_index.json")
            
        if base_class_name in IMG2IMG_MAPPING:
            mapped_class = IMG2IMG_MAPPING[base_class_name]
            if mapped_class != base_class_name:
                logger.info("  [WeeLLM] Mapping base pipeline '%s' to native image pipeline '%s'", base_class_name, mapped_class)
            return mapped_class
            
        if "Img2Img" in base_class_name or "Fill" in base_class_name or "Inpaint" in base_class_name:
            return base_class_name
            
        logger.warning("  [WeeLLM] No specific Img2Img mapping found for '%s', using base pipeline.", base_class_name)
        return base_class_name

    def __call__(self, *args, **kwargs):
        """
        Forward calls to the underlying diffusers pipeline while filtering
        out unsupported kwargs using introspection.
        """
        if kwargs.get("image") is None:
            raise ValueError(
                "[WeeLLM] WeeImageToImagePipeline requires an `image` argument. "
                "Use WeeTextToImagePipeline for text-only generation."
            )

        import inspect
        sig = inspect.signature(self._pipeline.__call__)
        supported_kwargs = set(sig.parameters.keys())
        
        # (Image dimension auto-fixing and _auto_resize disabling are now safely handled by WeeBasePipeline.__call__)

        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in supported_kwargs:
                filtered_kwargs[k] = v
            else:
                logger.debug("  [WeeLLM] Ignoring unsupported kwarg '%s' for %s", k, self._pipeline.__class__.__name__)
                
        return super().__call__(*args, **filtered_kwargs)

    def generate(self, prompt: str, image: Image.Image, **kwargs):
        """
        Convenience wrapper that calls the pipeline and returns the first image.

        Parameters
        ----------
        prompt:
            Text prompt for image generation.
        image:
            Input PIL Image.
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

        # Pass image to the pipeline natively through our overridden __call__
        out = self(prompt=prompt, image=image, generator=generator, **kwargs)
        if hasattr(out, "images"):
            return out.images[0]
        return out[0][0]


# Backward-compat alias so any code still importing WeeImagePipeline doesn't break.
WeeImagePipeline = WeeImageToImagePipeline
