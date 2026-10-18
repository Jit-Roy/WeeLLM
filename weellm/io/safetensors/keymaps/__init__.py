"""
weellm/io/safetensors/keymaps/__init__.py
Registry of safetensors key-map plugins.

Mirrors the GGUF keymap registry pattern (weellm/io/ggufs/keymaps/__init__.py)
but operates on safetensors weight_map keys instead of GGUF tensor names.

Each keymap plugin must define:
    NAME           : str            -- human-readable label for logging
    detect(keys)   : bool           -- return True if this map owns the checkpoint
    build_remap(keys) : dict        -- {src_key: dst_key} (1-to-1 flat mapping)
    postprocess_tensor(dst_key, tensor, src_key) : tensor  -- optional reshape/cast

Keymaps only handle key *renaming*.  Dequantization (comfy_quant / SVDQ) is
handled transparently by the seeker layer (comfy_dequant.py) before keymaps run.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any

logger = logging.getLogger("weellm")

from weellm.io.safetensors.keymaps.transformers.flux3 import Flux3SafetensorsKeyMap
from weellm.io.safetensors.keymaps.transformers.krea2 import Krea2SafetensorsKeyMap
from weellm.io.safetensors.keymaps.vaes.qwen_image_vae import QwenImageVAESafetensorsKeyMap

# Ordered: first match wins. Place more-specific detectors first.
_REGISTRY = [
    Flux3SafetensorsKeyMap,  # FLUX.3 DiT and variant exports (fp8r, nvfp4, etc.)
    Krea2SafetensorsKeyMap,   # model.diffusion_model.* (gonzalomo ComfyUI)
    QwenImageVAESafetensorsKeyMap,  # conv1/conv2 + decoder.upsamples.* (original Wan/Qwen VAE)
]


def build_remap_fn(
    weight_map_keys: List[str],
) -> Tuple[Optional[Callable[[str], str]], Any]:
    """
    Detect the checkpoint naming convention from the weight_map keys and return::

        (remap_fn, keymap_cls)

    ``remap_fn(src_key) -> dst_key``  maps a raw checkpoint key to its diffusers
    equivalent.  Returns the key unchanged if no mapping is defined.

    If no registered map matches, returns (None, None) — pass-through mode.
    """
    for keymap_cls in _REGISTRY:
        if keymap_cls.detect(weight_map_keys):
            remap_dict: Dict[str, str] = keymap_cls.build_remap(weight_map_keys)
            logger.info(
                "[SafetensorsKeymap] Detected format: %s — remapping to diffusers convention.",
                keymap_cls.NAME,
            )

            def _remap(key: str, _d=remap_dict) -> str:
                return _d.get(key, key)

            return _remap, keymap_cls

    return None, None


__all__ = [
    "build_remap_fn",
    "Flux3SafetensorsKeyMap",
    "Krea2SafetensorsKeyMap",
    "QwenImageVAESafetensorsKeyMap",
]
