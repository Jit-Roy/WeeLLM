"""
WeeLLM — Layer-streaming inference for large diffusion models.

Public API
----------
WeeTextToImagePipeline
    Pipeline for text-to-image generation.
    Use :meth:`WeeTextToImagePipeline.from_pretrained` to create a native
    diffusers pipeline with WeeLLM streamers injected.

WeeImageToImagePipeline
    Pipeline for image-to-image editing.

WeeVideoPipeline
    Pipeline for video generation.

Model streamers (for advanced / direct use):
    LazyVAEStreamer
    FluxTransformer2DModelStreamer, Flux2Transformer2DModelStreamer
    ZImageTransformer2DModelStreamer, SD3Transformer2DModelStreamer
    AuraFlowTransformer2DModelStreamer, CogView4Transformer2DModelStreamer
    HiDreamImageTransformer2DModelStreamer, Lumina2Transformer2DModelStreamer
    QwenImageTransformer2DModelStreamer, Ideogram4Transformer2DModelStreamer
    UNet2DConditionModelStreamer
"""

import logging

__version__ = "0.1.0"
__all__ = [
    # Core pipelines
    "WeeBasePipeline",
    "WeeTextToImagePipeline",
    "WeeImageToImagePipeline",
    "WeeVideoPipeline",
    # Backward-compat aliases
    "WeePipeline",
    "WeeImagePipeline",
    # VAE
    "LazyVAEStreamer",
    # Transformers
    "FluxTransformer2DModelStreamer",
    "Flux2Transformer2DModelStreamer",
    "ZImageTransformer2DModelStreamer",
    "SD3Transformer2DModelStreamer",
    "AuraFlowTransformer2DModelStreamer",
    "CogView4Transformer2DModelStreamer",
    "HiDreamImageTransformer2DModelStreamer",
    "Lumina2Transformer2DModelStreamer",
    "QwenImageTransformer2DModelStreamer",
    "Ideogram4Transformer2DModelStreamer",
    # UNet (SDXL / SD1.5)
    "UNet2DConditionModelStreamer",
]

# Add a NullHandler by default so library users don't get "No handlers" warnings.
# Applications that want output should configure their own handlers.
logging.getLogger("weellm").addHandler(logging.NullHandler())

from .weebasepipeline import WeeBasePipeline  # noqa: E402
from .weetexttoimagepipeline import WeeTextToImagePipeline, WeePipeline  # noqa: E402
from .weeimagetoimagepipeline import WeeImageToImagePipeline, WeeImagePipeline  # noqa: E402
from .weevideopipeline import WeeVideoPipeline  # noqa: E402

# VAE
from .models.vaes.lazy_vae import LazyVAEStreamer  # noqa: E402

# Transformers
from .models.transformers.flux_transformer_2d_model      import FluxTransformer2DModelStreamer  # noqa: E402
from .models.transformers.flux2_transformer_2d_model     import Flux2Transformer2DModelStreamer  # noqa: E402
from .models.transformers.z_image_transformer_2d_model   import ZImageTransformer2DModelStreamer  # noqa: E402
from .models.transformers.sd3_transformer_2d_model        import SD3Transformer2DModelStreamer  # noqa: E402
from .models.transformers.auraflow_transformer_2d_model  import AuraFlowTransformer2DModelStreamer  # noqa: E402
from .models.transformers.cogview4_transformer_2d_model  import CogView4Transformer2DModelStreamer  # noqa: E402
from .models.transformers.hidream_transformer_2d_model   import HiDreamImageTransformer2DModelStreamer  # noqa: E402
from .models.transformers.lumina2_transformer_2d_model   import Lumina2Transformer2DModelStreamer  # noqa: E402
from .models.transformers.qwen_image_transformer_2d_model import QwenImageTransformer2DModelStreamer  # noqa: E402
from .models.transformers.ideogram4_transformer          import Ideogram4Transformer2DModelStreamer  # noqa: E402

# UNet (SDXL / SD 1.5)
from .models.unets.unet_2d_condition_model import UNet2DConditionModelStreamer  # noqa: E402