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
        
        # Infer target dtype from other floating point tensors in result, default to bfloat16
        target_dtype = torch.bfloat16
        for k, v in result.items():
            if v.is_floating_point() and not any(suffix in k for suffix in ["weight_scale", "weight_s_rel", "weight_s_channel", "weight_codebook"]):
                target_dtype = v.dtype
                break
                
        math_dtype = torch.float32 if w.device.type == "cpu" else target_dtype
        
        is_asym = meta.get("format") == "asym_w4a8_int8"
        out_features = w.shape[0]
        in_features = w.shape[1] * 2 if is_asym else w.shape[1]
        
        out_tensor = torch.empty((out_features, in_features), device=w.device, dtype=target_dtype)
        
        chunk_size = 1024
        
        has_convrot = meta.get("convrot", False)
        if has_convrot:
            rot_size = meta.get("convrot_groupsize", 256)
            H_dtype = torch.float32 if w.device.type == "cpu" else target_dtype
            H = generate_hadamard(rot_size, device=str(w.device), dtype=H_dtype)
        else:
            rot_size = 256
            H = None
            
        if not is_asym:
            scale = None
            if prefix + "weight_scale" in result:
                scale = result[prefix + "weight_scale"].to(w.device)
                if scale.dim() == 1 and scale.size(0) == out_features:
                    scale = scale.unsqueeze(1)
                del result[prefix + "weight_scale"]
                
            for i in range(0, out_features, chunk_size):
                end_i = min(i + chunk_size, out_features)
                w_chunk = w[i:end_i].to(math_dtype)
                
                if scale is not None:
                    s_chunk = scale[i:end_i] if scale.shape[0] == out_features else scale
                    w_chunk = w_chunk * s_chunk.to(math_dtype)
                    
                if has_convrot:
                    w_reshaped = w_chunk.view(-1, rot_size)
                    H_math = H if w.device.type == "cpu" else H.to(math_dtype)
                    w_chunk = (w_reshaped @ H_math).view(end_i - i, in_features)
                    
                out_tensor[i:end_i] = w_chunk.to(target_dtype)
                
        else:
            w = w.view(torch.uint8)
            cb = result[prefix + "weight_codebook"].to(torch.float32)
            s_rel = result[prefix + "weight_s_rel"].to(torch.float32)
            s_chan = result[prefix + "weight_s_channel"].to(torch.float32)
            group_size = meta.get("group_size", 16)
            
            for i in range(0, out_features, chunk_size):
                end_i = min(i + chunk_size, out_features)
                w_chunk = w[i:end_i]
                
                v0 = (w_chunk & 0x0F).to(torch.long)
                v1 = ((w_chunk >> 4) & 0x0F).to(torch.long)
                w0_float = cb[v0]
                w1_float = cb[v1]
                
                w_float_chunk = torch.stack([w0_float, w1_float], dim=-1).view(end_i - i, -1)
                
                w_float_chunk = w_float_chunk.view(end_i - i, -1, group_size)
                w_float_chunk = w_float_chunk * s_rel[i:end_i].unsqueeze(-1)
                
                w_float_chunk = w_float_chunk.view(end_i - i, -1).round().clamp(-127, 127)
                
                w_float_chunk = w_float_chunk * s_chan[i:end_i].unsqueeze(1)
                
                w_float_chunk = w_float_chunk.to(math_dtype)
                
                if has_convrot:
                    w_reshaped = w_float_chunk.view(-1, rot_size)
                    H_math = H if w.device.type == "cpu" else H.to(math_dtype)
                    w_float_chunk = (w_reshaped @ H_math).view(end_i - i, in_features)
                    
                out_tensor[i:end_i] = w_float_chunk.to(target_dtype)
                
            del result[prefix + "weight_codebook"]
            del result[prefix + "weight_s_rel"]
            del result[prefix + "weight_s_channel"]
            
        del result[prefix + "comfy_quant"]
        result[prefix + "weight"] = out_tensor

    return result
