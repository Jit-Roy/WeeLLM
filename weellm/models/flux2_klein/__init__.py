"""
FLUX.2 Klein 4B -- layer-streaming inference package.

Public API
----------
    from weellm.models.flux2_klein import WeeFlux2KleinPipeline

    pipe = WeeFlux2KleinPipeline.from_pretrained(
        model_dir="flux2-klein-4b",
        device="cuda",
        dtype=torch.bfloat16,
    )
    image = pipe.generate(prompt="A sunset over mountains", height=512, width=512)
"""

from .pipeline import WeeFlux2KleinPipeline

__all__ = ["WeeFlux2KleinPipeline"]
