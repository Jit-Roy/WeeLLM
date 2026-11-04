"""
wan.py -- LoRA key mapping for Wan 2.1 / 2.2 and Wan Animate models.
Maps original checkpoint LoRA keys (from Wan training repos / PEFT) to Diffusers parameter names.
"""

from typing import Set, List, Tuple, Any

_WAN_RENAME_MAP = {
    ".self_attn.q": ".self_attn.to_q",
    ".self_attn.k": ".self_attn.to_k",
    ".self_attn.v": ".self_attn.to_v",
    ".self_attn.o": ".self_attn.to_out.0",
    ".cross_attn.q": ".cross_attn.to_q",
    ".cross_attn.k": ".cross_attn.to_k",
    ".cross_attn.v": ".cross_attn.to_v",
    ".cross_attn.o": ".cross_attn.to_out.0",
    ".cross_attn.k_img": ".cross_attn.add_k_proj",
    ".cross_attn.v_img": ".cross_attn.add_v_proj",
}


class WanLoRAKeyMap:
    NAME = "wan"

    @staticmethod
    def detect(lora_keys: Set[str]) -> bool:
        # Detect Wan LoRA keys containing blocks.*.self_attn or cross_attn
        for k in lora_keys:
            if "blocks." in k and (".self_attn." in k or ".cross_attn." in k):
                return True
        return False

    @staticmethod
    def build_pairs(lora_keys: Set[str]) -> List[Tuple[str, str, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()

        a_suffix = ".lora_A.weight"
        b_suffix = ".lora_B.weight"

        if not any(a_suffix in k for k in lora_keys):
            if any(".lora_down.weight" in k for k in lora_keys):
                a_suffix = ".lora_down.weight"
                b_suffix = ".lora_up.weight"

        for k in lora_keys:
            if a_suffix in k:
                bases.add(k.split(a_suffix)[0])

        for base in bases:
            a_key = f"{base}{a_suffix}"
            b_key = f"{base}{b_suffix}"
            if a_key not in lora_keys or b_key not in lora_keys:
                continue

            name = base
            for prefix in ("diffusion_model.", "base_model.model.", "transformer."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break

            # Apply Wan attention attribute remapping
            for orig_attr, diff_attr in _WAN_RENAME_MAP.items():
                if name.endswith(orig_attr):
                    name = name[:-len(orig_attr)] + diff_attr
                    break

            target_key = f"transformer.{name}.weight"
            pairs.append((a_key, b_key, [(target_key, None)]))

        return pairs
