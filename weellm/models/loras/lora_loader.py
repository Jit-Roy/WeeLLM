"""
Dynamic RAM-based LoRA Loader for WeeLLM MiniMax-H3 streaming.
Loads the Larry LoRA permanently into CPU RAM, and provides a method
to rapidly inject it into VRAM block modules on the fly.
"""

import os
import torch
import logging

logger = logging.getLogger("weellm")

class MiniMaxH3LoRALoader:
    def __init__(self, inner_dim: int = None, strength: float = 1.0):
        self.inner_dim = inner_dim  # will be auto-detected from file if None
        self.strength = strength
        self.entries = []
        self._load()

    def _larry_targets(self, name: str, b: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Map one reference-tree base name and its `lora_B` onto diffusers parameter key + row-transformed B."""
        if name.startswith("token_refiner.blocks."):
            target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
        elif name.startswith("blocks."):
            target = name.replace("blocks.", "transformer_blocks.", 1)
        else:
            target = name
        target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear")

        if target.endswith(".attn.qkv_proj"):
            prefix = target.removesuffix("qkv_proj")
            return [
                (f"{prefix}to_{kind}.weight", part.contiguous())
                for kind, part in zip(("q", "k", "v"), b.split(self.inner_dim, dim=0))
            ]
        if target.endswith(".mlp.fc1"):
            gate, value = b.chunk(2, dim=0)
            return [(target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", torch.cat([value, gate]).contiguous())]
        if target.endswith(".mlp.fc2"):
            return [(target.replace(".mlp.fc2", ".ff.net.2") + ".weight", b)]
        if target.endswith(".attn.out_proj"):
            return [(target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", b)]
        # `adaln_proj.linear` (block-level and the final `norm_out.linear`): identical row layout on both sides.
        return [(target + ".weight", b)]

    def _load(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        repo_id = os.environ.get("H3_LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
        filename = os.environ.get("H3_LORA", "minimax_h3_turbo_4step_ema_ckpt850.safetensors")
        
        logger.info(f"[LoRA] Downloading / Loading {repo_id}/{filename} into CPU RAM...")
        file_path = hf_hub_download(repo_id, filename)
        
        # Load into RAM
        lora = load_file(file_path)
        bases = sorted({key.rsplit(".lora_", 1)[0] for key in lora})
        
        # Auto-detect inner_dim from the first qkv_proj lora_B in the file.
        # lora_B for qkv_proj has shape [3 * inner_dim, rank], so inner_dim = B.shape[0] // 3
        if self.inner_dim is None:
            for name in bases:
                if name.endswith(".attn.qkv_proj"):
                    b = lora[f"{name}.lora_B.weight"]
                    self.inner_dim = b.shape[0] // 3
                    logger.info(f"[LoRA] Auto-detected inner_dim = {self.inner_dim} (from {name})")
                    break
            if self.inner_dim is None:
                raise RuntimeError("[LoRA] Could not auto-detect inner_dim: no .attn.qkv_proj key found in LoRA file.")
        
        for name in bases:
            a = lora[f"{name}.lora_A.weight"]
            b = lora[f"{name}.lora_B.weight"]
            self.entries.extend((key, a, b_part) for key, b_part in self._larry_targets(name, b))
            
        logger.info(f"[LoRA] Loaded {len(self.entries)} weight deltas into CPU RAM.")

    def apply_to_module(self, module: torch.nn.Module, shard_name: str):
        """
        Applies the LoRA delta to a specific block (shard) directly in VRAM.
        `shard_name` usually looks like "transformer.transformer_blocks.0"
        """
        # Find which LoRA entries belong to this shard
        # Note: self.entries keys don't have "transformer." prefix, they start with "transformer_blocks.N."
        
        # Determine the prefix to match in the LoRA dict
        # Diffusers keys in entries: "transformer_blocks.0.attn.to_q.weight"
        # shard_name might be "transformer_blocks.0" or "transformer.transformer_blocks.0"
        
        prefix = shard_name
        if prefix.startswith("transformer."):
            prefix = prefix[len("transformer."):]
        if prefix:
            prefix = prefix + "."
        
        params = dict(module.named_parameters())
        
        applied_count = 0
        for key, a, b in self.entries:
            if key.startswith(prefix):
                # local_key is what `module.named_parameters()` has, e.g. "attn.to_q.weight"
                local_key = key[len(prefix):]
                
                param = params.get(local_key)
                if param is not None:
                    # Both B and A are in RAM. Move them to the same device as the param (VRAM)
                    # and do the matrix multiplication on the GPU
                    with torch.no_grad():
                        delta = self.strength * (b.float() @ a.float())
                        param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                    applied_count += 1
                    
        if applied_count > 0:
            logger.debug(f"[LoRA] Applied {applied_count} LoRA tensors to {shard_name}")
