from typing import Set, List, Tuple, Any, Callable

class DiffusersFallbackLoRAKeyMap:
    NAME = "diffusers_fallback"

    @staticmethod
    def detect(lora_keys: Set[str]) -> bool:
        """Always return True. This acts as the fallback map."""
        return True

    @staticmethod
    def build_pairs(lora_keys: Set[str]) -> List[Tuple[str, str, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()
        
        # Determine suffixes based on what's available
        a_suffix = ".lora_A.weight"
        b_suffix = ".lora_B.weight"
        
        # Allow fallback for kohya/comfy style if present
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

            base_path = base
            # Generic stripping for diffusers format
            if base_path.startswith("diffusion_model."):
                base_path = "transformer." + base_path[len("diffusion_model."):]
            elif base_path.startswith("base_model.model."):
                base_path = "transformer." + base_path[len("base_model.model."):]
                
            targets = [(base_path + ".weight", None)]
            pairs.append((a_key, b_key, targets))
            
        return pairs
