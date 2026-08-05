"""
unet_streamer.py -- Hook-based block-streaming for UNet2DConditionModel (SDXL).

Strategy:
  - Resident on GPU: conv_in, time_embedding, add_embedding, conv_norm_out, conv_out
  - Streamed:        down_blocks, mid_block, up_blocks  (loaded just-in-time, evicted after)

Uses the same accelerate-based set_module_tensor_to_device pattern as Flux2Transformer2DModelStreamer.
"""

import torch
import torch.nn as nn
import types

from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from diffusers import UNet2DConditionModel

from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory


def _apply_state_dict(model: nn.Module, state_dict: dict, device: str, dtype: torch.dtype):
    """Write tensors into model parameters – handles meta -> real device."""
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            set_module_tensor_to_device(model, name, device, value=tensor, dtype=dtype)
        else:
            set_module_tensor_to_device(model, name, device, value=tensor)


def _evict_params(model: nn.Module, param_names: list):
    """Move named parameters back to meta device (free VRAM)."""
    for name in param_names:
        set_module_tensor_to_device(model, name, "meta")


def _patch_forward_input_device(model: nn.Module):
    """Move UNet forward inputs onto the model device before execution."""
    original_forward = model.forward

    def patched_forward(self_obj, *args, **kwargs):
        try:
            model_device = next(self_obj.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        args = list(args)
        for index, value in enumerate(args[:4]):
            if torch.is_tensor(value) and value.device != model_device:
                args[index] = value.to(model_device)

        for key, value in list(kwargs.items()):
            if torch.is_tensor(value) and value.device != model_device:
                kwargs[key] = value.to(model_device)

        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)


class UNet2DConditionModelStreamer:
    """
    Wraps UNet2DConditionModel for memory-efficient block streaming.
    Streams directly from original safetensors shards via SafetensorsLiveSeeker.
    Resident: conv_in, time_embedding, add_embedding, conv_norm_out, conv_out
    Streamed: down_blocks, mid_block, up_blocks
    """

    def __init__(
        self,
        model: UNet2DConditionModel,
        seeker,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
    ):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch

        # Build ordered list of (prefix, module) for all streamed blocks
        self._blocks: list[tuple[str, nn.Module]] = []
        for i, block in enumerate(model.down_blocks):
            self._blocks.append((f"down_blocks.{i}.", block))
        if hasattr(model, "mid_block") and model.mid_block is not None:
            self._blocks.append(("mid_block.", model.mid_block))
        for i, block in enumerate(model.up_blocks):
            self._blocks.append((f"up_blocks.{i}.", block))

        self._prefix_to_pos = {prefix: idx for idx, (prefix, _) in enumerate(self._blocks)}
        self._install_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self):
        for prefix, block in self._blocks:
            block._unet_prefix = prefix
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        prefix = module._unet_prefix
        keys = [k for k in self.seeker.weight_map if k.startswith(prefix)]
        sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
        _apply_state_dict(self.model, sd, self.device, self.dtype)
        module._unet_loaded_params = list(sd.keys())

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_unet_loaded_params", []))
        module._unet_loaded_params = []
        return output

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False
    ) -> "UNet2DConditionModelStreamer":
        import os
        path = os.path.join(model_dir, "unet")

        print("\nStep 1/3 -- Initializing LiveSeeker on UNet weights ...")
        seeker = get_seeker(path, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors.")

        print("\nStep 2/3 -- Instantiating UNet2DConditionModel on meta device ...")
        config = UNet2DConditionModel.load_config(os.path.join(path, "config.json"))
        with init_empty_weights():
            model = UNet2DConditionModel.from_config(config)
        model.eval()

        # Move any non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("\nStep 3/3 -- Loading resident UNet tensors to GPU ...")
        block_prefixes = ("down_blocks.", "mid_block.", "up_blocks.")
        resident_keys = [k for k in seeker.weight_map if not any(k.startswith(p) for p in block_prefixes)]
        resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd
        clean_memory(device)
        _patch_forward_input_device(model)
        report_memory("After resident load")

        print(f"\nInstalled {len(model.down_blocks)} down blocks, 1 mid block, {len(model.up_blocks)} up blocks for streaming.")
        streamer = cls(model, seeker, device, dtype, prefetch)
        print("\nUNetStreamer ready. Mode: Live Seek from original shards\n")
        return streamer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
