"""
Z-Image-Turbo -- layer-streaming inference package.

Public API
----------
    from weellm.models.z_image_turbo import WeeZImageTurboPipeline

    pipe = WeeZImageTurboPipeline.from_pretrained(
        model_dir="Z-Image-Turbo",
        device="cuda",
        dtype=torch.bfloat16,
    )
    image = pipe.generate(prompt="A futuristic city", height=512, width=512)
"""

from .pipeline import WeeZImageTurboPipeline

__all__ = ["WeeZImageTurboPipeline"]
