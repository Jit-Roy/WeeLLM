"""
Registry of all LoRA key-map plugins.
"""
import logging
from typing import Set, List, Tuple, Any, Callable

from weellm.io.safetensors.keymaps.loras.ltx2 import LTX2LoRAKeyMap
from weellm.io.safetensors.keymaps.loras.diffusers_fallback import DiffusersFallbackLoRAKeyMap
from weellm.io.safetensors.keymaps.loras.minimax import MiniMaxH3LoRAKeyMap

logger = logging.getLogger("weellm")

# A single LoRA target: (diffusers_key, slice_info)
# slice_info is None or (split_index, total_splits)
LoRATarget = Tuple[str, Any]

# A single paired entry: (a_file_key, b_file_key, list_of_targets)
LoRAPair = Tuple[str, str, List[LoRATarget]]

# Ordered: first match wins. Put more-specific detectors at the top.
_REGISTRY = [
    MiniMaxH3LoRAKeyMap,
    LTX2LoRAKeyMap,
    DiffusersFallbackLoRAKeyMap,  # Always keep fallback at the bottom
]

def build_lora_pairs(lora_keys: Set[str]) -> Tuple[List[LoRAPair], Any]:
    """
    Detect the LoRA naming convention from the key list and return the paired targets:
        (pairs, keymap_cls)
        
    pairs is a list of (a_key, b_key, [ (diffusers_key, slice_info), ... ])
    """
    for keymap_cls in _REGISTRY:
        if keymap_cls.detect(lora_keys):
            logger.info("[LoRA] Detected arch: %s — remapping keys.", keymap_cls.NAME)
            pairs = keymap_cls.build_pairs(lora_keys)
            return pairs, keymap_cls

    logger.debug("[LoRA] No key-map matched — empty pairs.")
    return [], None
