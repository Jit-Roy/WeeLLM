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
    quantization side-tensors (comfy_quant, SVDQ, NVFP4, scaled FP8) if the
    requested tensor is quantized.
    """
    expanded_keys = []
    for key in keys:
        expanded_keys.append(key)
        if key.endswith(".weight"):
            prefix = key[:-7]  # strip ".weight"

            # ── comfy_quant format (NF4, asym_w4a8) ───────────────────────────
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

            # ── NVFP4 / E2M1 group quantization (e.g. BFL / ModelOpt NVFP4) ───
            if (prefix + ".weight_scale_2") in weight_map:
                for suffix in ["weight_scale", "weight_scale_2", "input_scale"]:
                    candidate = prefix + "." + suffix
                    if candidate in weight_map:
                        expanded_keys.append(candidate)

            # ── Per-channel scaled FP8 (e.g. BFL / ComfyUI FP8) ───────────────
            elif (
                (prefix + ".weight_scale") in weight_map
                and meta_key not in weight_map
                and svdq_l1_key not in weight_map
            ):
                expanded_keys.append(prefix + ".weight_scale")

    # deduplicate but preserve order
    return list(dict.fromkeys(expanded_keys))


def process_fp8_and_nvfp4_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dequantize per-channel scaled FP8 and packed E2M1 (NVFP4) tensors."""
    target_dtype = torch.bfloat16
    for tensor in result.values():
        if tensor.is_floating_point() and tensor.dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            target_dtype = tensor.dtype
            break

    for weight_key in list(result):
        if not weight_key.endswith(".weight"):
            continue
        scale_key = weight_key[:-7] + ".weight_scale"
        if scale_key not in result:
            continue

        weight = result[weight_key]
        scales = result[scale_key]
        prefix = weight_key[:-7]

        # These are handled by the ComfyUI and SVDQ passes.
        if prefix + ".comfy_quant" in result or prefix + ".svdq_l1" in result:
            continue

        # FP8 per-output-channel export (scale is 1D matching weight row count).
        if weight.ndim == 2 and scales.ndim == 1 and scales.shape[0] == weight.shape[0]:
            result[weight_key] = weight.to(target_dtype) * scales.to(target_dtype).unsqueeze(1)
            del result[scale_key]
            continue

        # Packed E2M1 (NVFP4): 2 values per byte, group scale in weight_scale, global in weight_scale_2.
        scale2_key = weight_key[:-7] + ".weight_scale_2"
        if weight.ndim == 2 and weight.dtype == torch.uint8 and scales.ndim == 2 and scale2_key in result:
            packed = weight
            nibbles = torch.stack(
                ((packed & 0x0F), (packed >> 4) & 0x0F), dim=-1
            ).reshape(weight.shape[0], -1)
            magnitude = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                device=weight.device,
                dtype=torch.float32,
            )
            values = magnitude[(nibbles & 0x07).long()]
            values = torch.where(
                (nibbles & 0x08) != 0, -values, values
            )
            group_scales = scales.to(torch.float32).view(weight.shape[0], -1, 1)
            values = values.view(weight.shape[0], -1, 16)
            result[weight_key] = (
                values * group_scales * result[scale2_key].to(torch.float32).view(1, 1, 1)
            ).reshape(weight.shape[0], -1).to(target_dtype)
            del result[scale_key], result[scale2_key]
            input_scale_key = weight_key[:-7] + ".input_scale"
            result.pop(input_scale_key, None)

    return result


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

def _unpack_int4_weights(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 4-bit signed integers (INT4, two 4-bit values per uint8 byte) -> float32 (out, in)."""
    u8 = packed.view(torch.uint8)
    lo_u = (u8 & 0x0F)
    hi_u = ((u8 >> 4) & 0x0F)
    # Sign-extend 4-bit: values 8-15 → -8..-1
    lo = torch.where(lo_u >= 8, lo_u.to(torch.int16) - 16, lo_u.to(torch.int16)).to(torch.float32)
    hi = torch.where(hi_u >= 8, hi_u.to(torch.int16) - 16, hi_u.to(torch.int16)).to(torch.float32)
    out_rows, in_packed = u8.shape
    result = torch.empty(out_rows, in_packed * 2, dtype=torch.float32, device=packed.device)
    result[:, 0::2] = lo
    result[:, 1::2] = hi
    return result


def process_svdq_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Find SVDQ groups in the loaded tensors (detected by companion .svdq_l1 key),
    dequantize them, and remove the raw quantization tensors from the dict.
    Returns the same dict with .weight tensors replaced by their dequantized form.

    Supports both plain SVDQ and ConvRot-SVDQ (format "convrot_w4a4"), where the
    4-bit unpacked weights live in a Hadamard-rotated basis and must be un-rotated
    before adding the low-rank correction.
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

        # Read comfy_quant metadata to determine exact format
        cq_key = prefix + ".comfy_quant"
        has_convrot = False
        rot_size = 256
        if cq_key in result:
            meta_bytes = result[cq_key]
            if meta_bytes.dtype != torch.uint8:
                meta_bytes = meta_bytes.view(torch.uint8)
            try:
                meta = json.loads(bytes(meta_bytes.tolist()).decode("utf-8"))
                fmt = meta.get("format", "")
                # "convrot_w4a4" means weights were stored in a Hadamard-rotated
                # basis; we must un-rotate each row-group before reconstruction.
                has_convrot = "convrot" in fmt
                rot_size = int(meta.get("convrot_groupsize", 256))
            except Exception:
                pass

        # Infer target dtype from surrounding floating-point tensors
        target_dtype = torch.bfloat16
        for k, v in result.items():
            if v.is_floating_point() and k not in (s_key, l1_key, l2_key):
                target_dtype = v.dtype
                break

        w = result[w_key]
        scale = result[s_key]
        l1 = result[l1_key]
        l2 = result[l2_key]

        if has_convrot:
            # ConvRot-SVDQ: W_q is in the Hadamard-rotated basis.
            # Dequant: W_physical = (unpack(W_q) * scale[:, None] @ H) + L1 @ L2
            # where H is the normalised Hadamard matrix of size rot_size.
            device = w.device
            H = generate_hadamard(rot_size, device=str(device), dtype=torch.float32)
            w_unpacked = _unpack_int4_weights(w)          # float32 (out, in)
            w_scaled   = w_unpacked * scale.float().unsqueeze(1)  # (out, in)
            # Un-rotate in chunks to avoid building a huge intermediate tensor
            out_features, in_features = w_scaled.shape
            w_rot = torch.empty_like(w_scaled)
            for i in range(0, in_features, rot_size):
                end_i = min(i + rot_size, in_features)
                w_rot[:, i:end_i] = w_scaled[:, i:end_i] @ H[:end_i - i, :end_i - i]
            w_base = w_rot
        else:
            # Plain SVDQ: no rotation, direct scale + low-rank
            w_base = _unpack_int4_weights(w) * scale.float().unsqueeze(1)

        w_lr = l1.float() @ l2.float()
        result[w_key] = (w_base + w_lr).to(target_dtype)

        del result[s_key], result[l1_key], result[l2_key]
        if cq_key in result:
            del result[cq_key]

    return result


def process_comfy_quant_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Finds comfy_quant groups (NF4, asym_w4a8) in the loaded tensors, dequantizes them
    into standard .weight tensors, and removes the quantization metadata from the dictionary.
    """
    prefixes = []
    for k in result.keys():
        if k.endswith(".comfy_quant"):
            prefixes.append(k[:-12])  # strip ".comfy_quant"

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


def process_comfy_tensors(result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Master dequantization pipeline for loaded checkpoint tensors.
    Executes all format passes in order:
      1. SVDQ (gonzalomo-style 4-bit + low-rank correction, optional Hadamard ConvRot)
      2. Per-channel scaled FP8 and NVFP4 / E2M1 group-quantized weights
      3. ComfyUI comfy_quant metadata weights (NF4, asym_w4a8)
    """
    result = process_svdq_tensors(result)
    result = process_fp8_and_nvfp4_tensors(result)
    result = process_comfy_quant_tensors(result)
    return result
