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

Transformer streamers
Text encoder streamers
Unet streamers
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
    # IO — memory & placement
    "place_tensors",
    "evict_module",
    "pin_module_to_cpu",
    "get_seeker",
    "override_weights_path",
    "SafetensorsBase",
    "SafetensorsDiskSeeker",
    "SafetensorsRAMSeeker",
    "GGUFSeeker",
    "dequantize_tensor",
    # VAE
    "AutoencoderKL",
    "AutoencoderKLMiniMaxH3Streamer",
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
    "QwenImage21Transformer2DModelStreamer",
    "Ideogram4Transformer2DModelStreamer",
    "ErnieImageTransformer2DModelStreamer",
    "Krea2Transformer2DModelStreamer",
    "LongCatImageTransformer2DModelStreamer",
    "LTX2ConnectorsStreamer",
    "LTX2VideoTransformer3DModelStreamer",
    "MiniMaxH3Transformer3DModelStreamer",
    # Text Encoders
    "ChatGLMModelStreamer",
    "CLIPTextModelStreamer",
    "Gemma2ModelStreamer",
    "Gemma4UnifiedForConditionalGenerationStreamer",
    "GlmModelStreamer",
    "LlamaForCausalLMStreamer",
    "Mistral3ForConditionalGenerationStreamer",
    "Mistral3ModelStreamer",
    "Qwen2_5_VLForConditionalGenerationStreamer",
    "Qwen3ForCausalLMStreamer",
    "Qwen3VLForConditionalGenerationStreamer",
    "Qwen3VLModelStreamer",
    "T5EncoderModelStreamer",
    "UMT5EncoderModelStreamer",
    # UNet (SDXL / SD1.5)
    "UNet2DConditionModelStreamer",
]

# Add a NullHandler by default so library users don't get "No handlers" warnings.
# Applications that want output should configure their own handlers.
logging.getLogger("weellm").addHandler(logging.NullHandler())

from .pipelines.weebasepipeline import WeeBasePipeline  # noqa: E402
from .pipelines.image.weetexttoimagepipeline import WeeTextToImagePipeline, WeePipeline  # noqa: E402
from .pipelines.image.weeimagetoimagepipeline import WeeImageToImagePipeline, WeeImagePipeline  # noqa: E402
from .pipelines.video.weevideopipeline import WeeVideoPipeline  # noqa: E402

# IO — memory & tensor placement 
from .io.memory import place_tensors, evict_module, pin_module_to_cpu  # noqa: E402
from .io.seeker import get_seeker, override_weights_path  # noqa: E402
from .io.safetensors.safetensors_base import SafetensorsBase  # noqa: E402
from .io.safetensors.disk_seek import SafetensorsDiskSeeker  # noqa: E402
from .io.safetensors.ram_seek import SafetensorsRAMSeeker  # noqa: E402
from .io.ggufs.gguf_seek import GGUFSeeker  # noqa: E402
from .io.ggufs.gguf_dequant import dequantize_tensor  # noqa: E402

# VAE
from .models.vaes.autoencoder_kl import AutoencoderKL  # noqa: E402
from .models.vaes.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3Streamer  # noqa: E402

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
from .models.transformers.qwen_image_2_1_transformer_2d_model import QwenImage21Transformer2DModelStreamer  # noqa: E402
from .models.transformers.ideogram4_transformer          import Ideogram4Transformer2DModelStreamer  # noqa: E402
from .models.transformers.ernie_image_transformer_2d_model import ErnieImageTransformer2DModelStreamer  # noqa: E402
from .models.transformers.krea2_transformer_2d_model import Krea2Transformer2DModelStreamer  # noqa: E402
from .models.transformers.longcat_transformer_2d_model import LongCatImageTransformer2DModelStreamer  # noqa: E402
from .models.transformers.ltx2_connectors import LTX2ConnectorsStreamer  # noqa: E402
from .models.transformers.ltx2_video_transformer_3d_model import LTX2VideoTransformer3DModelStreamer  # noqa: E402
from .models.transformers.minimax_h3_transformer_3d_model import MiniMaxH3Transformer3DModelStreamer  # noqa: E402

# Text Encoders
from .models.text_encoders.chatglm_model import ChatGLMModelStreamer  # noqa: E402
from .models.text_encoders.clip_text_model import CLIPTextModelStreamer  # noqa: E402
from .models.text_encoders.gemma2_model import Gemma2ModelStreamer  # noqa: E402
from .models.text_encoders.gemma4_unified_for_conditional_generation import Gemma4UnifiedForConditionalGenerationStreamer  # noqa: E402
from .models.text_encoders.glm_model import GlmModelStreamer  # noqa: E402
from .models.text_encoders.llama_for_causal_lm import LlamaForCausalLMStreamer  # noqa: E402
from .models.text_encoders.mistral3_for_conditional_generation import Mistral3ForConditionalGenerationStreamer  # noqa: E402
from .models.text_encoders.mistral3_model import Mistral3ModelStreamer  # noqa: E402
from .models.text_encoders.qwen2_5_vl_for_conditional_generation import Qwen2_5_VLForConditionalGenerationStreamer  # noqa: E402
from .models.text_encoders.qwen3_for_causal_lm import Qwen3ForCausalLMStreamer  # noqa: E402
from .models.text_encoders.qwen3_vl_for_conditional_generation import Qwen3VLForConditionalGenerationStreamer  # noqa: E402
from .models.text_encoders.qwen3_vl_model import Qwen3VLModelStreamer  # noqa: E402
from .models.text_encoders.t5_encoder_model import T5EncoderModelStreamer  # noqa: E402
from .models.text_encoders.umt5_encoder_model import UMT5EncoderModelStreamer  # noqa: E402

# UNet (SDXL / SD 1.5)
from .models.unets.unet_2d_condition_model import UNet2DConditionModelStreamer  # noqa: E402