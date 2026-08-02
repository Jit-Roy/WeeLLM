"""
WeeLLM -- Layer-streaming inference for large diffusion models.

Fits multi-billion-parameter diffusion models under 4 GB VRAM and 8 GB RAM
by streaming one transformer layer at a time from disk, using the same
hook-based technique pioneered by AirLLM for language models.

Supported models
----------------
    flux2-klein   FLUX.2 Klein 4B  (4 GB VRAM, 8 GB RAM)

Quick start
-----------
    from weellm import get_pipeline
    import torch

    Pipeline = get_pipeline("flux2-klein")
    pipe = Pipeline.from_pretrained("flux2-klein-4b", device="cuda", dtype=torch.bfloat16)
    image = pipe.generate("A sunset over mountains", height=512, width=512)
    image.save("output.png")
"""

from .registry import get_pipeline, list_models, register_model, MODEL_REGISTRY
from .core.base_pipeline import BasePipeline
from .core.base_streamer import BaseStreamer

# Convenience shortcut: import the most common pipeline directly
from .models.flux2_klein import WeeFlux2KleinPipeline

__version__ = "0.1.0"

__all__ = [
    # Registry API
    "get_pipeline",
    "list_models",
    "register_model",
    "MODEL_REGISTRY",
    # Base classes (for model implementers)
    "BasePipeline",
    "BaseStreamer",
    # Built-in pipelines
    "WeeFlux2KleinPipeline",
]
