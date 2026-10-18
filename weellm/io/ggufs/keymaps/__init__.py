"""
weellm/io/ggufs/keymaps/__init__.py -- Registry of all GGUF key-map plugins.

Adding a new architecture:
  1. Create a new file in this directory (e.g. myarch.py).
  2. Define a class with three members:
       NAME     : str            -- human-readable label for logging
       detect() : (keys, arch) -> bool   -- return True if this map owns the file
       build_remap() : (keys) -> dict    -- return flat {gguf_key: [(diffusers_key, slice_info), ...]}
  3. Import and append the class to _REGISTRY below.

The registry is evaluated in ORDER — place more-specific detectors before
more-general ones to avoid false matches.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from weellm.io.ggufs.keymaps.text_encoders.t5     import T5GGUFKeyMap
from weellm.io.ggufs.keymaps.text_encoders.llama  import LlamaGGUFKeyMap
from weellm.io.ggufs.keymaps.text_encoders.gemma4  import Gemma4GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.flux   import FluxGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.flux2  import Flux2GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.flux3  import Flux3GGUFKeyMap
from weellm.io.ggufs.keymaps.text_encoders.glm    import GLMGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.sd3    import SD3GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.sdxl   import SDXLGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.sd15   import SD15GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.krea2  import Krea2GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.zimage import ZImageGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.minimax_h3 import MiniMaxH3GGUFKeyMap
from weellm.io.ggufs.keymaps.text_encoders.qwen3vl import Qwen3VLGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.ltx25 import LTX25GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.qwen_image_21 import QwenImage21GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.longcat import LongCatGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.auraflow import AuraFlowGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.cogview4 import CogView4GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.ernie_image import ErnieImageGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.hidream import HiDreamGGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.ideogram4 import Ideogram4GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.lumina2 import Lumina2GGUFKeyMap
from weellm.io.ggufs.keymaps.transformers.qwen_image import QwenImageGGUFKeyMap

logger = logging.getLogger("weellm")

# Ordered: first match wins.  Put more-specific detectors at the top.
_REGISTRY = [
    T5GGUFKeyMap,       # enc.blk.*  — must come before llama (no overlap, but explicit ordering)
    GLMGGUFKeyMap,      # GLM text encoder: token_embd + blk.* + ffn_gate
    Gemma4GGUFKeyMap,   # multi_modal_projector.* + *.layer_scalar (already diffusers naming)
    LlamaGGUFKeyMap,    # blk.*
    Flux2GGUFKeyMap,    # Flux.2 Klein: 8 double + 24 single blocks
    Flux3GGUFKeyMap,    # Flux.3 Image Klein: content_mode_blocks + single_blocks
    LongCatGGUFKeyMap,  # MUST come before FluxGGUFKeyMap: uses double_blocks.* but NO vec_in/guidance_in
    AuraFlowGGUFKeyMap, # double_layers.* + w2q naming
    FluxGGUFKeyMap,     # double_blocks.*
    SD3GGUFKeyMap,      # joint_blocks.*
    SDXLGGUFKeyMap,     # model.diffusion_model.* + label_emb / transformer_blocks.9
    SD15GGUFKeyMap,     # model.diffusion_model.* (no SDXL markers)
    Krea2GGUFKeyMap,    # txtfusion.* or blocks.0.attn.qknorm.*
    ZImageGGUFKeyMap,   # context_refiner.* / noise_refiner.*
    Qwen3VLGGUFKeyMap,  # visual.blocks.* + model.layers.* (unsloth Qwen3VL TE GGUF)
    MiniMaxH3GGUFKeyMap,# blocks.* (MiniMax H3 checkpoint convention)
    LTX25GGUFKeyMap,    # ltxv native keys
    QwenImage21GGUFKeyMap,
    CogView4GGUFKeyMap,
    ErnieImageGGUFKeyMap,
    HiDreamGGUFKeyMap,
    Ideogram4GGUFKeyMap,
    Lumina2GGUFKeyMap,
    QwenImageGGUFKeyMap,
]


def build_remap_fn(gguf_keys: List[str], arch: str = "unknown") -> tuple[Callable[[str], List], Any]:
    """
    Detect the GGUF naming convention from the key list and return a tuple::

        (remap_fn, keymap_cls)

    ``remap_fn(gguf_key) -> [(diffusers_key, slice_info), ...]``
    ``slice_info`` is either ``None`` (no splitting needed) or
    ``(split_index, total_splits)`` for fused QKV tensors.

    If no registered map matches, a pass-through function and None is returned.
    """
    for keymap_cls in _REGISTRY:
        if keymap_cls.detect(gguf_keys, arch):
            remap_dict: Dict[str, Any] = keymap_cls.build_remap(gguf_keys)
            logger.info("[GGUFSeeker] Detected arch: %s — remapping to Diffusers convention.", keymap_cls.NAME)

            def _remap(name: str, _d=remap_dict) -> List:
                return _d.get(name, [(name, None)])

            return _remap, keymap_cls

    # No match — pass every key through unchanged
    logger.debug("[GGUFSeeker] No key-map matched (arch=%s) — using pass-through.", arch)
    return lambda name: [(name, None)], None


__all__ = [
    "build_remap_fn",
    "T5GGUFKeyMap",
    "LlamaGGUFKeyMap",
    "Gemma4GGUFKeyMap",
    "GLMGGUFKeyMap",
    "FluxGGUFKeyMap",
    "Flux2GGUFKeyMap",
    "Flux3GGUFKeyMap",
    "SD3GGUFKeyMap",
    "SDXLGGUFKeyMap",
    "SD15GGUFKeyMap",
    "Krea2GGUFKeyMap",
    "ZImageGGUFKeyMap",
    "Qwen3VLGGUFKeyMap",
    "MiniMaxH3GGUFKeyMap",
    "LTX25GGUFKeyMap",
    "QwenImage21GGUFKeyMap",
    "LongCatGGUFKeyMap",
    "AuraFlowGGUFKeyMap",
    "CogView4GGUFKeyMap",
    "ErnieImageGGUFKeyMap",
    "HiDreamGGUFKeyMap",
    "Ideogram4GGUFKeyMap",
    "Lumina2GGUFKeyMap",
    "QwenImageGGUFKeyMap",
]