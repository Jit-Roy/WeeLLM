"""
weellm/gguf_keymaps/__init__.py -- Registry of all GGUF key-map plugins.

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

from weellm.io.ggufs.keymaps.text_encoders.t5     import T5KeyMap
from weellm.io.ggufs.keymaps.text_encoders.llama  import LlamaKeyMap
from weellm.io.ggufs.keymaps.text_encoders.gemma4  import Gemma4KeyMap
from weellm.io.ggufs.keymaps.transformers.flux   import FluxKeyMap
from weellm.io.ggufs.keymaps.transformers.flux2  import Flux2KeyMap
from weellm.io.ggufs.keymaps.text_encoders.glm    import GLMKeyMap
from weellm.io.ggufs.keymaps.transformers.sd3    import SD3KeyMap
from weellm.io.ggufs.keymaps.transformers.sdxl   import SDXLKeyMap
from weellm.io.ggufs.keymaps.transformers.sd15   import SD15KeyMap
from weellm.io.ggufs.keymaps.transformers.krea2  import Krea2KeyMap
from weellm.io.ggufs.keymaps.transformers.zimage import ZImageKeyMap
from weellm.io.ggufs.keymaps.transformers.minimax_h3 import MiniMaxH3KeyMap
from weellm.io.ggufs.keymaps.text_encoders.qwen3vl import Qwen3VLKeyMap
from weellm.io.ggufs.keymaps.transformers.ltx25 import LTX25KeyMap
from weellm.io.ggufs.keymaps.transformers.qwen_image_21 import QwenImage21KeyMap

logger = logging.getLogger("weellm")

# Ordered: first match wins.  Put more-specific detectors at the top.
_REGISTRY = [
    T5KeyMap,       # enc.blk.*  — must come before llama (no overlap, but explicit ordering)
    GLMKeyMap,      # GLM text encoder: token_embd + blk.* + ffn_gate
    Gemma4KeyMap,   # multi_modal_projector.* + *.layer_scalar (already diffusers naming)
    LlamaKeyMap,    # blk.*
    Flux2KeyMap,    # Flux.2 Klein: 8 double + 24 single blocks
    FluxKeyMap,     # double_blocks.*
    SD3KeyMap,      # joint_blocks.*
    SDXLKeyMap,     # model.diffusion_model.* + label_emb / transformer_blocks.9
    SD15KeyMap,     # model.diffusion_model.* (no SDXL markers)
    Krea2KeyMap,    # txtfusion.* or blocks.0.attn.qknorm.*
    ZImageKeyMap,   # context_refiner.* / noise_refiner.*
    Qwen3VLKeyMap,  # visual.blocks.* + model.layers.* (unsloth Qwen3VL TE GGUF)
    MiniMaxH3KeyMap,# blocks.* (MiniMax H3 checkpoint convention)
    LTX25KeyMap,     # ltxv native keys
    QwenImage21KeyMap,
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
    "T5KeyMap",
    "LlamaKeyMap",
    "Gemma4KeyMap",
    "GLMKeyMap",
    "FluxKeyMap",
    "Flux2KeyMap",
    "SD3KeyMap",
    "SDXLKeyMap",
    "SD15KeyMap",
    "Krea2KeyMap",
    "ZImageKeyMap",
    "Qwen3VLKeyMap",
    "MiniMaxH3KeyMap",
    "LTX25KeyMap",
    "QwenImage21KeyMap",
]