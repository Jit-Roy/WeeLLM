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
    comfy_quant and svdq side-tensors if the requested tensor is quantized.
    """
    expanded_keys = []
    for key in keys:
        expanded_keys.append(key)
        if key.endswith(".weight"):
            prefix = key[:-7] # strip ".weight"
            # ── comfy_quant format ────────────────────────────────────────
            meta_key = prefix + ".comfy_quant"
            if meta_key in weight_map:
                expanded_keys.append(meta_key)
                for suffix in ["weight_codebook", "weight_s_rel", "weight_s_channel", "weight_scale"]:
                    candidate = prefix + "." + suffix
                    if candidate in weight_map:
                        expanded_keys.append(candidate)
            # ── SVDQ format (gonzalomo-style nibble-packed 4-bit + low-rank) ──
            svdq_l1_key = prefix + ".svdq_l1"
            if svdq_l1_key in weight_map:
                for suffix in ["weight_scale", "svdq_l1", "svdq_l2"]:
                    candidate = prefix + "." + suffix
                    if candidate in weight_map:
                        expanded_keys.append(candidate)
    # deduplicate but preserve order
    return list(dict.fromkeys(expanded_keys))


# ---------------------------------------------------------------------------
# SVDQ dequantization
#
# Format (gonzalomo / ComfyUI SVDQ):
#   .weight      I8  (out, in//2)  — 4-bit nibbles packed, 2 per byte
#   .weight_scale F32 (out,)       — per-row scale
#   .svdq_l1     BF16 (out, rank)  — low-rank correction factor 1
#   .svdq_l2     BF16 (rank, in)   — low-rank correction factor 2
#
# Reconstruction:  W = unpack_nibbles(I8) * scale[:,None] + L1 @ L2
# ---------------------------------------------------------------------------

def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack nibble-packed I8 (out, in//2) → float32 (out, in)."""
    u8 = packed.view(torch.uint8)
    lo_u = (u8 & 0x0F)
    hi_u = ((u8 >> 4) & 0x0F)
    # Sign-extend 4-bit: values 8-15 → -8..-1
    lo = torch.where(lo_u >= 8, lo_u.to(torch.int16) - 16, lo_u.to(torch.int16)).to(torch.float32)
    hi = torch.where(hi_u >= 8, hi_u.to(torch.int16) - 16, hi_u.to(torch.int16)).to(torch.float32)
    out_rows, in_packed = u8.shape
    result = torch.empty(out_rows, in_packed * 2, dtype=torch.float32)
    result[:, 0::2] = lo
    result[:, 1::2] = hi
    return result


def _dequant_svdq(
    weight: torch.Tensor,    # I8  (out, in//2)
    scale: torch.Tensor,     # F32 (out,)
    svdq_l1: torch.Tensor,   # BF16 (out, rank)
    svdq_l2: torch.Tensor,   # BF16 (rank, in)
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a full BF16/F32 matrix from SVDQ components."""
    w_base = _unpack_nibbles(weight) * scale.float().unsqueeze(1)
    w_lr   = svdq_l1.float() @ svdq_l2.float()
    return (w_base + w_lr).to(target_dtype)


def process_svdq_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Find SVDQ groups in the loaded tensors (detected by companion .svdq_l1 key),
    dequantize them, and remove the raw quantization tensors from the dict.
    Returns the same dict with .weight tensors replaced by their dequantized form.
    """
    prefixes = [
        k[:-len(".svdq_l1")]
        for k in list(result.keys())
        if k.endswith(".svdq_l1")
    ]
    for prefix in prefixes:
        w_key  = prefix + ".weight"
        s_key  = prefix + ".weight_scale"
        l1_key = prefix + ".svdq_l1"
        l2_key = prefix + ".svdq_l2"
        if not all(k in result for k in (w_key, s_key, l1_key, l2_key)):
            continue

        # Infer target dtype from surrounding floating-point tensors
        target_dtype = torch.bfloat16
        for k, v in result.items():
            if v.is_floating_point() and k not in (s_key, l1_key, l2_key):
                target_dtype = v.dtype
                break

        result[w_key] = _dequant_svdq(
            result[w_key], result[s_key], result[l1_key], result[l2_key],
            target_dtype=target_dtype,
        )
        del result[s_key], result[l1_key], result[l2_key]
        # comfy_quant metadata blob is also present but not needed after dequant
        cq_key = prefix + ".comfy_quant"
        if cq_key in result:
            del result[cq_key]

    return result


def process_comfy_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Finds comfy_quant groups in the loaded tensors, dequantizes them into standard
    .weight tensors, and removes the quantization metadata from the dictionary.
    Also handles SVDQ groups (gonzalomo-style 4-bit + low-rank).
    """
    # ── SVDQ pass (must run before comfy_quant pass) ──────────────────────────
    result = process_svdq_tensors(result)

    # ── Legacy comfy_quant pass ───────────────────────────────────────────────
    # Find all comfy_quant prefixes
    prefixes = []
    for k in result.keys():
        if k.endswith(".comfy_quant"):
            prefixes.append(k[:-12]) # strip ".comfy_quant"
            
    for prefix in prefixes:
        meta_bytes = result[prefix + ".comfy_quant"]
        if meta_bytes.dtype != torch.uint8:
            meta_bytes = meta_bytes.view(torch.uint8)
        meta = json.loads(bytes(meta_bytes.tolist()).decode("utf-8"))
        
        w = result[prefix + ".weight"]
        
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
            if prefix + ".weight_scale" in result:
                scale = result[prefix + ".weight_scale"].to(w.device)
                if scale.dim() == 1 and scale.size(0) == out_features:
                    scale = scale.unsqueeze(1)
                del result[prefix + ".weight_scale"]
                
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
            cb = result[prefix + ".weight_codebook"].to(torch.float32)
            s_rel = result[prefix + ".weight_s_rel"].to(torch.float32)
            s_chan = result[prefix + ".weight_s_channel"].to(torch.float32)
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
                
            del result[prefix + ".weight_codebook"]
            del result[prefix + ".weight_s_rel"]
            del result[prefix + ".weight_s_channel"]
            
        del result[prefix + ".comfy_quant"]
        result[prefix + ".weight"] = out_tensor

    return result
