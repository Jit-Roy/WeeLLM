"""
gguf_dequant.py -- Pure-PyTorch GGUF block dequantization kernels for WeeLLM.

Converts raw quantized GGUF byte-arrays back into standard FP16/BF16 tensors
using only native PyTorch bitwise and arithmetic operations.  No custom CUDA
extension or C++ compilation required.

Supported quantization types
-----------------------------
Non-quantized : F32, F16, BF16
Legacy        : Q8_0, Q5_1, Q5_0, Q4_1, Q4_0
K-Quants      : Q2_K, Q3_K, Q4_K, Q5_K, Q6_K  (most common on HuggingFace)
IQ-Quants     : IQ4_NL, IQ4_XS

Ported from the Apache-2.0 licensed ComfyUI-GGUF project by city96:
https://github.com/city96/ComfyUI-GGUF
"""

import gguf
import torch
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Quant types that can be used directly as torch tensors without dequantization.
TORCH_COMPATIBLE_QTYPES = frozenset({
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
})


def is_quantized(qtype) -> bool:
    """Return True if *qtype* requires dequantization before GPU use."""
    return qtype not in TORCH_COMPATIBLE_QTYPES


def to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def to_uint16(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def split_block_dims(blocks: torch.Tensor, *args) -> tuple:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# ---------------------------------------------------------------------------
# Non-quantized / full-weight handlers
# ---------------------------------------------------------------------------

def _dequant_BF16(blocks: torch.Tensor, block_size: int, type_size: int,
                  dtype=None) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# ---------------------------------------------------------------------------
# Legacy quant handlers
# ---------------------------------------------------------------------------

def _dequant_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequant_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    m  = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m


def _dequant_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dequant_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d  = d.view(torch.float16).to(dtype)
    m  = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _dequant_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# ---------------------------------------------------------------------------
# K-Quant helpers & handlers
# ---------------------------------------------------------------------------

QK_K         = 256
K_SCALE_SIZE = 12


def _get_scale_min(scales: torch.Tensor):
    n_blocks = scales.shape[0]
    scales   = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc  = torch.cat([d & 0x3F,  (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min_ = torch.cat([m & 0x3F, (m_d >> 4)   | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), min_.reshape((n_blocks, 8))


def _dequant_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d      = d.view(torch.float16).to(dtype)
    d      = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql     = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql     = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh     = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh     = (qh & 0x03).reshape((n_blocks, -1, 32))
    q      = (ql | (qh << 4)).to(torch.int8) - 32
    q      = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dequant_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d    = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d    = (d * sc).reshape((n_blocks, -1, 1))
    dm   = (dmin * m).reshape((n_blocks, -1, 1))
    ql   = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh   = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql   = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh   = (qh & 0x01).reshape((n_blocks, -1, 32))
    q    = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequant_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d    = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d    = (d * sc).reshape((n_blocks, -1, 1))
    dm   = (dmin * m).reshape((n_blocks, -1, 1))
    qs   = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs   = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequant_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d    = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales  = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales  = (scales.to(torch.int8) - 32)
    dl      = (d * scales).reshape((n_blocks, 16, 1))
    ql      = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh      = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql      = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh      = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q       = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def _dequant_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d    = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl   = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml   = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs   = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs   = qs.reshape((n_blocks, QK_K // 16, 16))
    qs   = dl * qs - ml
    return qs.reshape((n_blocks, -1))


# ---------------------------------------------------------------------------
# IQ-Quant handlers
# ---------------------------------------------------------------------------

_KVALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)


def _dequant_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs    = split_block_dims(blocks, 2)
    d        = d.view(torch.float16).to(dtype)
    qs       = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs       = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)
    kvalues  = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs       = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return d * qs


def _dequant_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d        = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)
    shift_a  = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b  = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03
    scales   = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl       = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    qs       = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs       = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F
    kvalues  = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs       = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    return (dl * qs).reshape((n_blocks, -1))


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DEQUANT_FUNCTIONS = {
    gguf.GGMLQuantizationType.BF16:   _dequant_BF16,
    gguf.GGMLQuantizationType.Q8_0:   _dequant_Q8_0,
    gguf.GGMLQuantizationType.Q5_1:   _dequant_Q5_1,
    gguf.GGMLQuantizationType.Q5_0:   _dequant_Q5_0,
    gguf.GGMLQuantizationType.Q4_1:   _dequant_Q4_1,
    gguf.GGMLQuantizationType.Q4_0:   _dequant_Q4_0,
    gguf.GGMLQuantizationType.Q6_K:   _dequant_Q6_K,
    gguf.GGMLQuantizationType.Q5_K:   _dequant_Q5_K,
    gguf.GGMLQuantizationType.Q4_K:   _dequant_Q4_K,
    gguf.GGMLQuantizationType.Q3_K:   _dequant_Q3_K,
    gguf.GGMLQuantizationType.Q2_K:   _dequant_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: _dequant_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: _dequant_IQ4_XS,
}


def dequantize_tensor(
    raw_data: torch.Tensor,
    qtype: "gguf.GGMLQuantizationType",
    shape: tuple,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize a GGUF tensor from its raw quantized bytes to a standard
    PyTorch floating-point tensor.

    Parameters
    ----------
    raw_data:
        1-D uint8 tensor containing the raw quantized block bytes as read
        from the GGUF file via memory-mapping.
    qtype:
        The GGUF quantization type (e.g. ``gguf.GGMLQuantizationType.Q4_K``).
    shape:
        The original unquantized tensor shape (rows, cols, ...) as stored in
        the ``comfy.gguf.orig_shape.*`` metadata field, or derived from the
        reversed GGUF shape.
    dtype:
        The target PyTorch dtype for the output tensor (e.g. ``torch.bfloat16``).
        Defaults to ``torch.float16`` when None.

    Returns
    -------
    torch.Tensor
        A standard floating-point tensor with the given *shape* and *dtype*.
        Ready for direct use by PyTorch nn.Linear and other ops.
    """
    if dtype is None:
        dtype = torch.float16

    # F32 / F16 — just reinterpret
    if qtype == gguf.GGMLQuantizationType.F32:
        return raw_data.view(torch.float32).reshape(shape).to(dtype)
    if qtype == gguf.GGMLQuantizationType.F16:
        return raw_data.view(torch.float16).reshape(shape).to(dtype)

    # BF16 and quantized types — use block-level dequantizers
    if qtype in _DEQUANT_FUNCTIONS:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        dequant_fn            = _DEQUANT_FUNCTIONS[qtype]

        rows     = raw_data.reshape((-1, raw_data.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // type_size
        blocks   = rows.reshape((n_blocks, type_size))
        result   = dequant_fn(blocks, block_size, type_size, dtype)
        return result.reshape(shape).to(dtype)

    # Final fallback — use gguf's own numpy-based path (slow but universal)
    import warnings
    import numpy as np
    warnings.warn(
        f"WeeLLM: No fast PyTorch dequantizer for {qtype}. "
        "Falling back to slow numpy path.",
        RuntimeWarning,
        stacklevel=2,
    )
    np_arr = gguf.quants.dequantize(raw_data.cpu().numpy(), qtype)
    return torch.from_numpy(np_arr).reshape(shape).to(dtype)
