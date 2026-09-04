"""
weeimagepipeline.py -- WeePipeline for Image-to-Image and Image Editing.
"""

from typing import Optional
import torch
import logging
from PIL import Image

from weellm.pipeline import WeeBasePipeline

logger = logging.getLogger("weellm")

IMG2IMG_MAPPING = {
    "StableDiffusionPipeline": "StableDiffusionImg2ImgPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusion3Pipeline": "StableDiffusion3Img2ImgPipeline",
    "LongCatImagePipeline": "LongCatImageEditPipeline",
    "HiDreamImagePipeline": "HiDreamImageEditingPipeline",
}

class WeeImagePipeline(WeeBasePipeline):
    """
    Image-to-Image WeePipeline.
    """
    
    @classmethod
    def _get_diffusers_pipeline_class(cls, index: dict) -> str:
        base_class_name = index.get("_class_name")
        if not base_class_name:
            raise ValueError("No _class_name found in model_index.json")
            
        if base_class_name in IMG2IMG_MAPPING:
            mapped_class = IMG2IMG_MAPPING[base_class_name]
            logger.info("  [WeeLLM] Mapping base pipeline '%s' to native image pipeline '%s'", base_class_name, mapped_class)
            return mapped_class
            
        logger.warning("  [WeeLLM] No specific Img2Img mapping found for '%s', using base pipeline.", base_class_name)
        return base_class_name

    def __call__(self, *args, **kwargs):
        """
        Forward calls to the underlying diffusers pipeline while filtering
        out unsupported kwargs using introspection.
        """
        import inspect
        sig = inspect.signature(self._pipeline.__call__)
        supported_kwargs = set(sig.parameters.keys())
        
        # Auto-fix image dimensions (diffusers pipelines often require multiples of 64
        # and tend to squish images to default 1024x1024 if width/height are omitted)
        if "image" in kwargs and kwargs["image"] is not None:
            img = kwargs["image"]
            if isinstance(img, Image.Image):
                w, h = img.size
                
                # Flux Fill supports multiples of 16, others strictly require 64
                divisor = 16 if self._pipeline.__class__.__name__ == "FluxFillPipeline" else 64
                new_w = max(divisor, (w // divisor) * divisor)
                new_h = max(divisor, (h // divisor) * divisor)
                
                if w != new_w or h != new_h:
                    logger.info("  [WeeLLM] Auto-resizing input image from %dx%d to %dx%d (must be multiple of %d)", w, h, new_w, new_h, divisor)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    kwargs["image"] = img
                
                # Force pipeline to respect image dimensions instead of falling back to default squares
                if "height" not in kwargs and "height" in supported_kwargs:
                    kwargs["height"] = new_h
                if "width" not in kwargs and "width" in supported_kwargs:
                    kwargs["width"] = new_w

        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in supported_kwargs:
                filtered_kwargs[k] = v
            else:
                logger.debug("  [WeeLLM] Ignoring unsupported kwarg '%s' for %s", k, self._pipeline.__class__.__name__)
                
        return self._pipeline(*args, **filtered_kwargs)

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
