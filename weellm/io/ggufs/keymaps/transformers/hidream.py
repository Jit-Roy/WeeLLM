"""
hidream.py -- GGUF key map for HiDream (ComfyUI format).

ComfyUI stores HiDream weights natively in Hugging Face Diffusers format.
The only difference in ComfyUI GGUF files is the "model.diffusion_model." prefix.
This keymap strips that prefix so WeeLLM can load them directly.

Verified via: comfyui/comfy/model_detection.py::convert_diffusers_mmdit()
HiDream is NOT present in that function, confirming native diffusers naming.
"""
from typing import Any, Dict, List


class HiDreamKeyMap:
    """
    Strips the ComfyUI "model.diffusion_model." prefix from HiDream GGUF files.
    """
    NAME = "hidream"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # Comfy prefix present AND model-specific signature key exists
        has_comfy_prefix = any(k.startswith("model.diffusion_model.") for k in gguf_keys)
        has_sig = any("model.diffusion_model.caption_projection." in k for k in gguf_keys)
        return has_comfy_prefix and has_sig

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        return {gguf_key: gguf_key.replace("model.diffusion_model.", "") for gguf_key in gguf_keys}
