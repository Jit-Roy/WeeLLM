"""
memory.py -- Unified tensor placement and eviction helpers for WeeLLM streamers.

All weight-loading and VRAM-eviction across the entire codebase goes through
exactly two functions defined here:

    place_tensors(model, state_dict, device, dtype)
        Move a dict of tensors onto a model that was created on the meta device.

    evict_module(module)
        Move every parameter and buffer inside a module back to the meta device,
        freeing VRAM immediately.

No other code should import ``set_module_tensor_to_device`` directly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device


def place_tensors(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> None:
    """
    Write *state_dict* tensors into *model* parameters (meta → real device).

    Floating-point tensors are cast to *dtype*.
    Integer / bool tensors (embedding indices, masks) are moved as-is.

    Parameters
    ----------
    model:
        The nn.Module whose parameters will be filled (typically created with
        ``init_empty_weights()`` so its parameters are on the meta device).
    state_dict:
        Mapping of fully-qualified parameter name → tensor value.
    device:
        Target device string, e.g. ``"cuda"`` or ``"cpu"``.
    dtype:
        Target floating-point dtype, e.g. ``torch.bfloat16``.
    """
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)
        else:
            set_module_tensor_to_device(model, name, device, value=tensor)


def evict_module(module: nn.Module) -> int:
    """
    Move every parameter and buffer inside *module* back to the meta device.

    This is used by all streaming post-hooks to free VRAM as soon as a
    transformer block finishes its forward pass.

    Uses ``set_module_tensor_to_device`` (the same accelerate helper used when
    loading weights) so PyTorch's internal bookkeeping is correct and GPU
    memory is actually released.  Direct ``param.data = ...`` assignment is
    intentionally avoided — it creates a zero-sized junk tensor that PyTorch
    still keeps allocated.

    Parameters
    ----------
    module:
        Any nn.Module (e.g. a single transformer block or an entire model).

    Returns
    -------
    int — number of tensors evicted.
    """
    evicted = 0

    for name, param in list(module.named_parameters(recurse=True)):
        dev = getattr(param, "device", None)
        if dev is not None and dev.type != "meta":
            try:
                set_module_tensor_to_device(module, name, "meta")
            except Exception:
                pass
            evicted += 1

    for name, buf in list(module.named_buffers(recurse=True)):
        dev = getattr(buf, "device", None)
        if dev is not None and dev.type != "meta":
            try:
                set_module_tensor_to_device(module, name, "meta")
            except Exception:
                pass

    return evicted
