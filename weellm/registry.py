"""
registry.py -- Model registry for WeeLLM.

Maps short model identifiers (used on the CLI with --model) to their
pipeline classes.  Supports aliases so users can type natural names.

Adding a new model
------------------
1. Create  weellm/models/<your_model>/pipeline.py
           (subclass weellm.core.BasePipeline)
2. Add an entry below:

       from .models.your_model import LightYourModelPipeline
       register_model("your-model", LightYourModelPipeline)
"""

from __future__ import annotations

from typing import Dict, List, Type

from .core.base_pipeline import BasePipeline

# ---------------------------------------------------------------------------
# Registry store
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type[BasePipeline]] = {}


def register_model(name: str, pipeline_cls: Type[BasePipeline]) -> None:
    """Register a pipeline class under one or more names / aliases."""
    MODEL_REGISTRY[name] = pipeline_cls


def get_pipeline(name: str) -> Type[BasePipeline]:
    """
    Retrieve a pipeline class by model name.

    Raises
    ------
    ValueError
        If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = list_models()
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available models: {available}\n"
            f"Register new models in weellm/registry.py."
        )
    return MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """Return a sorted list of all registered model names."""
    return sorted(set(MODEL_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Built-in model registrations
# ---------------------------------------------------------------------------

def _register_builtin_models() -> None:
    from .models.flux2_klein import WeeFlux2KleinPipeline
    from .models.z_image_turbo import WeeZImageTurboPipeline
    from .models.sdxl import WeeSDXLPipeline
    from .models.sd15 import WeeSD15Pipeline
    from .models.flux_schnell import WeeFluxSchnellPipeline

    # FLUX.2 Klein 4B
    register_model("flux2-klein",  WeeFlux2KleinPipeline)
    register_model("flux2_klein",  WeeFlux2KleinPipeline)   # underscore alias

    # Z-Image-Turbo
    register_model("z-image-turbo",  WeeZImageTurboPipeline)
    register_model("z_image_turbo",  WeeZImageTurboPipeline)  # underscore alias

    # Stable Diffusion XL
    register_model("sdxl", WeeSDXLPipeline)

    # Stable Diffusion 1.5 / 2.x
    register_model("sd15",  WeeSD15Pipeline)
    register_model("sd1.5", WeeSD15Pipeline)   # dot alias

    # FLUX.1-schnell / FLUX.1-dev
    register_model("flux-schnell", WeeFluxSchnellPipeline)
    register_model("flux1-schnell", WeeFluxSchnellPipeline)  # long alias

    # ── Add future models here ──────────────────────────────────────────
    # from .models.sd35  import LightSD35Pipeline
    # register_model("sd35",  LightSD35Pipeline)
    # ────────────────────────────────────────────────────────────────────


_register_builtin_models()
