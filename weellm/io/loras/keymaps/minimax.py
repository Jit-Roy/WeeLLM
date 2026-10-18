from typing import Set, List, Tuple, Any

class MiniMaxH3LoRAKeyMap:
    NAME = "minimax_h3"

    @staticmethod
    def detect(lora_keys: Set[str]) -> bool:
        # MiniMax LoRAs usually have "token_refiner" or just "blocks"
        # We can also check inner_dim if needed, but checking for typical keys is fine
        for k in lora_keys:
            if k.startswith("token_refiner.blocks.") or k.startswith("blocks.") or ".mlp.fc1" in k:
                return True
        return False

    @staticmethod
    def build_pairs(lora_keys: Set[str]) -> List[Tuple[str, str, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()
        for k in lora_keys:
            if ".lora_A." in k:
                bases.add(k.split(".lora_A.")[0])

        for base in bases:
            a_key = f"{base}.lora_A.weight"
            b_key = f"{base}.lora_B.weight"
            if a_key not in lora_keys or b_key not in lora_keys:
                continue

            # Strip standard diffusers / peft / comfyui prefixes
            name = base
            for prefix in ("diffusion_model.", "transformer.", "base_model.model."):
                if name.startswith(prefix):
                    name = name[len(prefix):]

            if name.startswith("token_refiner.blocks."):
                target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
            elif name.startswith("blocks."):
                target = name.replace("blocks.", "transformer_blocks.", 1)
            else:
                target = name
            target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear")

            targets = []
            if target.endswith(".attn.qkv_proj"):
                prefix = target.removesuffix("qkv_proj")
                # We need to slice B into 3 pieces: to_q, to_k, to_v
                targets.append((f"{prefix}to_q.weight", (0, 3)))
                targets.append((f"{prefix}to_k.weight", (1, 3)))
                targets.append((f"{prefix}to_v.weight", (2, 3)))
            elif target.endswith(".mlp.fc1"):
                # We need to split into 2 pieces and reorder them: (value, gate) -> (gate, value)
                # This requires a special slice info or custom logic.
                # In GenericLazyLoRALoader, we can define slice_info = ("chunk_reorder", 2, [1, 0])
                targets.append((target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", ("chunk_reorder", 2, [1, 0])))
            elif target.endswith(".mlp.fc2"):
                targets.append((target.replace(".mlp.fc2", ".ff.net.2") + ".weight", None))
            elif target.endswith(".attn.out_proj"):
                targets.append((target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", None))
            else:
                targets.append((target + ".weight", None))

            pairs.append((a_key, b_key, targets))

        return pairs
