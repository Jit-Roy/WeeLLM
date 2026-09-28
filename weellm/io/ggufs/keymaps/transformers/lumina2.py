"""
lumina2.py -- GGUF key map for Lumina2 (ComfyUI format).

ComfyUI stores Lumina2 weights natively in Hugging Face Diffusers format.
The only difference in ComfyUI GGUF files is the "model.diffusion_model." prefix.
This keymap strips that prefix so WeeLLM can load them directly.

Verified via: comfyui/comfy/model_detection.py::convert_diffusers_mmdit()
Lumina2 is NOT present in that function, confirming native diffusers naming.
"""
from typing import Any, Dict, List


class Lumina2KeyMap:
    """
    Strips the ComfyUI "model.diffusion_model." prefix from Lumina2 GGUF files.
    """
    NAME = "lumina2"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        # Comfy prefix present AND model-specific signature key exists
        has_comfy_prefix = any(k.startswith("model.diffusion_model.") for k in gguf_keys)
        has_sig = any("model.diffusion_model.cap_embedder." in k for k in gguf_keys)
        return has_comfy_prefix and has_sig

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        return {gguf_key: gguf_key.replace("model.diffusion_model.", "") for gguf_key in gguf_keys}
