"""
weellm/io/safetensors/keymaps/loras/fibo2.py
LoRA key map for Bria Fibo 2.0 / Fibo 2.0 Turbo LoRA format.
"""

from typing import Set, List, Tuple, Any

class Fibo2LoRAKeyMap:
    NAME = "fibo2"

    @staticmethod
    def detect(lora_keys: Set[str]) -> bool:
        has_layers = any("diffusion_model.layers." in k or "layers.0." in k for k in lora_keys)
        has_perceiver = any("perceiver" in k for k in lora_keys)
        return has_layers and has_perceiver

    @staticmethod
    def build_pairs(lora_keys: Set[str]) -> List[Tuple[str, Any, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()

        a_suffix = ".lora_down.weight"
        b_suffix = ".lora_up.weight"
        if not any(a_suffix in k for k in lora_keys):
            if any(".lora_A.weight" in k for k in lora_keys):
                a_suffix = ".lora_A.weight"
                b_suffix = ".lora_B.weight"

        for k in lora_keys:
            if a_suffix in k:
                bases.add(k.split(a_suffix)[0])

        for base in sorted(bases):
            a_key = f"{base}{a_suffix}"
            b_key = f"{base}{b_suffix}"
            if a_key not in lora_keys or b_key not in lora_keys:
                continue

            path = base
            if path.startswith("diffusion_model."):
                path = path[len("diffusion_model."):]

            # Fused QKV
            if path.endswith(".attention.qkv"):
                b = "transformer." + path[:-len(".attention.qkv")] + ".attention"
                targets = [
                    (f"{b}.to_q.weight", (0, 3)),
                    (f"{b}.to_k.weight", (1, 3)),
                    (f"{b}.to_v.weight", (2, 3)),
                ]
            elif path.endswith(".attention.out"):
                b = "transformer." + path[:-len(".attention.out")] + ".attention"
                targets = [(f"{b}.to_out.0.weight", None)]
            elif path.endswith(".feed_forward.w13"):
                b = "transformer." + path[:-len(".feed_forward.w13")] + ".feed_forward"
                targets = [
                    (f"{b}.w1.weight", (0, 2)),
                    (f"{b}.w3.weight", (1, 2)),
                ]
            elif path.endswith(".feed_forward.w2"):
                b = "transformer." + path[:-len(".feed_forward.w2")] + ".feed_forward"
                targets = [(f"{b}.w2.weight", None)]
            elif path.endswith(".cross_attention.to_kv"):
                b = "transformer." + path[:-len(".cross_attention.to_kv")] + ".cross_attention"
                targets = [
                    (f"{b}.to_k.weight", (0, 2)),
                    (f"{b}.to_v.weight", (1, 2)),
                ]
            elif path.endswith(".cross_attention.to_out"):
                b = "transformer." + path[:-len(".cross_attention.to_out")] + ".cross_attention"
                targets = [(f"{b}.to_out.0.weight", None)]
            elif path.endswith(".self_attention.to_kv"):
                b = "transformer." + path[:-len(".self_attention.to_kv")] + ".self_attention"
                targets = [
                    (f"{b}.to_k.weight", (0, 2)),
                    (f"{b}.to_v.weight", (1, 2)),
                ]
            elif path.endswith(".self_attention.to_out"):
                b = "transformer." + path[:-len(".self_attention.to_out")] + ".self_attention"
                targets = [(f"{b}.to_out.0.weight", None)]
            elif path.endswith(".ffn.w13"):
                b = "transformer." + path[:-len(".ffn.w13")] + ".ffn"
                targets = [
                    (f"{b}.w1.weight", (0, 2)),
                    (f"{b}.w3.weight", (1, 2)),
                ]
            else:
                targets = [(f"transformer.{path}.weight", None)]

            pairs.append((a_key, b_key, targets))

        # Direct diff keys (.diff and .diff_b)
        for k in sorted(lora_keys):
            if k.endswith(".diff_b"):
                raw_path = k[:-len(".diff_b")]
                if raw_path.startswith("diffusion_model."):
                    raw_path = raw_path[len("diffusion_model."):]
                target = f"transformer.{raw_path}.bias"
                pairs.append((k, None, [(target, None)]))
            elif k.endswith(".diff"):
                raw_path = k[:-len(".diff")]
                if raw_path.startswith("diffusion_model."):
                    raw_path = raw_path[len("diffusion_model."):]
                if raw_path.endswith(".k_norm"):
                    raw_path = raw_path[:-len(".k_norm")] + ".norm_k"
                elif raw_path.endswith(".q_norm"):
                    raw_path = raw_path[:-len(".q_norm")] + ".norm_q"
                target = f"transformer.{raw_path}.weight"
                pairs.append((k, None, [(target, None)]))

        return pairs
