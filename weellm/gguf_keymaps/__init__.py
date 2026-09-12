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

from weellm.gguf_keymaps.t5     import T5KeyMap
from weellm.gguf_keymaps.llama  import LlamaKeyMap
from weellm.gguf_keymaps.flux   import FluxKeyMap
from weellm.gguf_keymaps.sdxl   import SDXLKeyMap
from weellm.gguf_keymaps.sd15   import SD15KeyMap
from weellm.gguf_keymaps.krea2  import Krea2KeyMap
from weellm.gguf_keymaps.zimage import ZImageKeyMap
from weellm.gguf_keymaps.minimax_h3 import MiniMaxH3KeyMap
from weellm.gguf_keymaps.qwen3vl import Qwen3VLKeyMap

logger = logging.getLogger("weellm")

# Ordered: first match wins.  Put more-specific detectors at the top.
_REGISTRY = [
    T5KeyMap,       # enc.blk.*  — must come before llama (no overlap, but explicit ordering)
    LlamaKeyMap,    # blk.*
    FluxKeyMap,     # double_blocks.*
    SDXLKeyMap,     # model.diffusion_model.* + label_emb / transformer_blocks.9
    SD15KeyMap,     # model.diffusion_model.* (no SDXL markers)
    Krea2KeyMap,    # txtfusion.* or blocks.0.attn.qknorm.*
    ZImageKeyMap,   # context_refiner.* / noise_refiner.*
    Qwen3VLKeyMap,  # visual.blocks.* + model.layers.* (unsloth Qwen3VL TE GGUF)
    MiniMaxH3KeyMap,# blocks.* (MiniMax H3 checkpoint convention)
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
    "FluxKeyMap",
    "SDXLKeyMap",
    "SD15KeyMap",
    "Krea2KeyMap",
    "ZImageKeyMap",
    "Qwen3VLKeyMap",
    "MiniMaxH3KeyMap",
]