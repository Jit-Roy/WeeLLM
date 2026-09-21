import json
import torch
from typing import Dict

from functools import lru_cache

@lru_cache(maxsize=4)
def generate_hadamard(n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Generates an n x n Hadamard matrix using ConvRot's h4 base."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < n:
        h = torch.kron(h, h4)
        current_size *= 4
    return h / (n ** 0.5)

def expand_comfy_keys(keys: list[str], weight_map: Dict[str, str]) -> list[str]:
    """
    Given a list of requested keys, expands them to include all necessary
    comfy_quant side-tensors if the requested tensor is quantized.
    """
    expanded_keys = []
    for key in keys:
        expanded_keys.append(key)
        if key.endswith(".weight"):
            prefix = key[:-7] # strip "weight"
            meta_key = prefix + "comfy_quant"
            if meta_key in weight_map:
                expanded_keys.append(meta_key)
                for suffix in ["weight_codebook", "weight_s_rel", "weight_s_channel", "weight_scale"]:
                    if prefix + suffix in weight_map:
                        expanded_keys.append(prefix + suffix)
    # deduplicate but preserve order somewhat
    return list(dict.fromkeys(expanded_keys))

def process_comfy_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Finds comfy_quant groups in the loaded tensors, dequantizes them into standard
    .weight tensors, and removes the quantization metadata from the dictionary.
    """
    # Find all comfy_quant prefixes
    prefixes = []
    for k in result.keys():
        if k.endswith(".comfy_quant"):
            prefixes.append(k[:-11]) # strip "comfy_quant"
            
    for prefix in prefixes:
        meta_bytes = result[prefix + "comfy_quant"]
        if meta_bytes.dtype != torch.uint8:
            meta_bytes = meta_bytes.view(torch.uint8)
        meta = json.loads(bytes(meta_bytes.tolist()).decode("utf-8"))
        
        w = result[prefix + "weight"]
        
        if meta.get("format") != "asym_w4a8_int8":
            # For linear quant formats (e.g., fp8 or int8)
            w_float = w.to(torch.float32)
            if "weight_scale" in result:
                scale = result[prefix + "weight_scale"].to(torch.float32)
                if scale.dim() == 1 and scale.size(0) == w_float.shape[0]:
                    scale = scale.unsqueeze(1)
                w_float = w_float * scale
                del result[prefix + "weight_scale"]
        else:
            w = w.view(torch.uint8)
            cb = result[prefix + "weight_codebook"].to(torch.float32)
            s_rel = result[prefix + "weight_s_rel"].to(torch.float32)
            s_chan = result[prefix + "weight_s_channel"].to(torch.float32)
            
            # 1 & 2 & 3. Unpack 4-bit values and lookup codebook directly to save memory
            v0 = (w & 0x0F).to(torch.long)
            v1 = ((w >> 4) & 0x0F).to(torch.long)
            w0_float = cb[v0]
            w1_float = cb[v1]
            w_float = torch.stack([w0_float, w1_float], dim=-1).view(w.shape[0], -1)
            
            # 4. Apply Group Scales (s_rel) using broadcasting instead of repeat_interleave
            group_size = meta.get("group_size", 16)
            w_float = w_float.view(w_float.shape[0], -1, group_size)
            w_float = w_float * s_rel.unsqueeze(-1)
            
            # Match comfy-kitchen exactly by rounding to integers and clamping to int8 range
            w_float = w_float.view(w.shape[0], -1).round().clamp(-127, 127)
            
            # 5. Apply Channel Scales (s_chan)
            w_float = w_float * s_chan.unsqueeze(1)
            
            del result[prefix + "weight_codebook"]
            del result[prefix + "weight_s_rel"]
            del result[prefix + "weight_s_channel"]
        
        # 6. Apply Hadamard Rotation (ConvRot)
        if meta.get("convrot", False):
            rot_size = meta.get("convrot_groupsize", 256)
            H = generate_hadamard(rot_size, device=str(w_float.device), dtype=w_float.dtype)
            out_features, in_features = w_float.shape
            w_reshaped = w_float.view(-1, rot_size)
            w_float = (w_reshaped @ H).view(out_features, in_features)
            
        # Cleanup and save
        del result[prefix + "comfy_quant"]
        result[prefix + "weight"] = w_float

    return result
