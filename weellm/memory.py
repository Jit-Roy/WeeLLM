"""
memory.py -- Unified tensor placement and eviction helpers for WeeLLM streamers.

All weight-loading and VRAM-eviction across the entire codebase goes through
exactly two functions defined here:

    place_tensors(model, state_dict, device, dtype)
        Move a dict of tensors onto a model that was created on the meta device.

    evict_module(module)
        Move every parameter and buffer inside a module back to the meta device,
        freeing VRAM immediately.

    pin_module_to_cpu(model, attr_path)
        Monkey-patch an embedding module so its forward pass executes on the CPU,
        saving massive VRAM for text encoder embeddings.

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
    skip_errors: bool = False,
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
    skip_errors:
        If True, silently skip keys that don't exist in the model rather than
        raising an error. Useful when checkpoint names don't perfectly align
        with the model's attribute names.
    """
    for name, tensor in state_dict.items():
        try:
            # Handle Conv2d (1x1) <-> Linear shape mismatches for Diffusers architectures
            try:
                obj = model
                for attr in name.split("."):
                    obj = getattr(obj, attr)
                if obj.shape != tensor.shape:
                    if len(tensor.shape) == 4 and len(obj.shape) == 2 and tensor.shape[2:] == (1, 1):
                        tensor = tensor.squeeze(-1).squeeze(-1)
                    elif len(tensor.shape) == 2 and len(obj.shape) == 4 and obj.shape[2:] == (1, 1):
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)
            except Exception:
                pass

            if tensor.is_floating_point():
                set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)
            else:
                set_module_tensor_to_device(model, name, device, value=tensor)
        except (AttributeError, ValueError) as exc:
            if skip_errors:
                import logging
                logging.getLogger("weellm").debug("Skipping tensor %s: %s", name, exc)
            else:
                raise


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


def pin_module_to_cpu(model: nn.Module, attr_path: str) -> None:
    """
    Monkey-patch an embedding module so its forward pass executes on CPU.
    
    This is used to route massive text encoder embeddings (e.g., 2.5GB for GLM)
    through system RAM without moving them to VRAM.
    
    Parameters
    ----------
    model:
        The root nn.Module.
    attr_path:
        The dot-separated path to the sub-module (e.g., ``"model.embed_tokens"``).
    """
    import logging
    logger = logging.getLogger("weellm")
    
    mod = model
    for attr in attr_path.split("."):
        if not hasattr(mod, attr):
            logger.warning("Could not find %s in %s for CPU pinning.", attr, attr_path)
            return
        mod = getattr(mod, attr)
        
    original_forward = mod.forward
    
    def cpu_forward(input_ids, *args, **kwargs):
        input_device = input_ids.device
        if input_device.type != "cpu":
            input_ids = input_ids.cpu()
        
        out = original_forward(input_ids, *args, **kwargs)
        
        if input_device.type != "cpu":
            out = out.to(input_device)
        return out
        
    mod.forward = cpu_forward
    logger.debug("Pinned %s to run on CPU.", attr_path)
