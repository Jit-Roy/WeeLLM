"""
auto.py -- Auto-router for WeeLLM.

Automatically detects the model architecture from the huggingface model_index.json
and routes it to the appropriate streaming pipeline class.
"""

import json
from pathlib import Path
from typing import Union

import torch

from .registry import MODEL_REGISTRY

# Map from HuggingFace pipeline class names (in model_index.json) to our internal registry keys.
# Note: FluxPipeline is ambiguous (used by both FLUX.1-schnell and FLUX.2-klein).
# We resolve this by checking the transformer config class inside _resolve_flux_variant().
AUTO_PIPELINE_MAPPING = {
    # Flux models
    "FluxPipeline": "flux-schnell",
    "Flux2KleinPipeline": "flux2-klein",

    # Z-Image-Turbo
    "ZImagePipeline": "z-image-turbo",

    # Stable Diffusion 1.x / 2.x
    "StableDiffusionPipeline": "sd15",

    # Stable Diffusion XL models
    "StableDiffusionXLPipeline": "sdxl",

    # Stable Diffusion 3.x / 3.5
    "StableDiffusion3Pipeline": "sd35",
}

class WeePipeline:
    """
    Auto-detects model architecture and loads the correct WeeLLM streaming pipeline.
    
    Example:
        from weellm import WeePipeline
        pipe = WeePipeline.from_pretrained("path/to/flux2-klein-4b")
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        from .core.utils import resolve_model_path
        
        model_path = resolve_model_path(str(model_id_or_path))
        index_path = model_path / "model_index.json"
        
        if not index_path.exists():
            print("WARNING: model_index.json not found, defaulting to flux2-klein architecture.")
            target_model = "flux2-klein"
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            hf_class = index.get("_class_name", "")
            if hf_class not in AUTO_PIPELINE_MAPPING:
                raise ValueError(
                    f"Unsupported model architecture: '{hf_class}'.\n"
                    f"Currently WeeLLM supports: {list(AUTO_PIPELINE_MAPPING.keys())}"
                )
            else:
                target_model = AUTO_PIPELINE_MAPPING[hf_class]
                print(f"AutoRouter: Detected '{hf_class}' -> Routing to WeeLLM '{target_model}' pipeline.")
            
        pipeline_cls = MODEL_REGISTRY[target_model]
        return pipeline_cls.from_pretrained(
            model_dir=model_path,
            device=device,
            dtype=dtype,
            **kwargs
        )
