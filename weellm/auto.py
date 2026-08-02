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

# Map from HuggingFace pipeline class names (in model_index.json) to our internal registry keys
AUTO_PIPELINE_MAPPING = {
    # Currently we route Flux-like models to our custom Klein pipeline.
    # Note: If a user tries to load a standard FLUX model (with T5/CLIP), 
    # it will fail in the text_encoder_streamer since we expect Qwen3.
    # When we add standard FLUX support, we can map this dynamically!
    "FluxPipeline": "flux2-klein",
    "Flux2KleinPipeline": "flux2-klein",
    
    # Future architectures:
    # "StableDiffusionXLPipeline": "sdxl",
    # "StableDiffusion3Pipeline": "sd35",
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
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        model_path = Path(model_dir)
        index_path = model_path / "model_index.json"
        
        if not index_path.exists():
            # If there's no model_index.json, fallback to the default (flux2-klein)
            print("WARNING: model_index.json not found, defaulting to flux2-klein architecture.")
            target_model = "flux2-klein"
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            hf_class = index.get("_class_name", "")
            if hf_class not in AUTO_PIPELINE_MAPPING:
                raise ValueError(
                    f"Unsupported model architecture: '{hf_class}'.\n"
                    f"Currently WeeLLM supports architectures: {list(AUTO_PIPELINE_MAPPING.keys())}"
                )
            
            target_model = AUTO_PIPELINE_MAPPING[hf_class]
            print(f"AutoRouter: Detected '{hf_class}' -> Routing to WeeLLM '{target_model}' pipeline.")
            
        pipeline_cls = MODEL_REGISTRY[target_model]
        return pipeline_cls.from_pretrained(
            model_dir=model_path,
            device=device,
            dtype=dtype,
            **kwargs
        )
