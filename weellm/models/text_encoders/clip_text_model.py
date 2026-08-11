"""
clip_streamer.py -- Hook-based layer-streaming for CLIPTextModel / CLIPTextModelWithProjection.

Strategy:
  - Resident on GPU: embeddings, final_layer_norm, text_projection (small, always needed)
  - Streamed:        encoder.layers[0..N]  (loaded just-in-time, evicted after)

Uses the same accelerate-based set_module_tensor_to_device pattern as FluxStreamer.

Key naming:
  - safetensors file may use "text_model.encoder.layers.X.*" prefix
  - model object may use "encoder.layers.X.*" (flattened, newer transformers)
  - We track key_strip to translate between the two
"""

import torch
import torch.nn as nn
import types

from accelerate import init_empty_weights
from weellm.utils import default_dtype
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory
from weellm.memory import place_tensors, evict_module


def _patch_forward_input_device(model: nn.Module, target_device: str = "cuda"):
    """Move CLIP text inputs onto the model device before forward runs."""
    original_forward = model.forward
    model_device = torch.device(target_device)

    def patched_forward(self_obj, *args, **kwargs):

        args = list(args)
        if args and torch.is_tensor(args[0]) and args[0].device != model_device:
            args[0] = args[0].to(model_device)

        for key in ("input_ids", "attention_mask", "position_ids", "inputs_embeds"):
            value = kwargs.get(key)
            if torch.is_tensor(value) and value.device != model_device:
                kwargs[key] = value.to(model_device)

        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)


class CLIPTextModelStreamer:
    """
    Wraps CLIPTextModel or CLIPTextModelWithProjection for layer-by-layer streaming.
    Resident: embeddings, final_layer_norm, text_projection
    Streamed:  encoder layers  (loaded per-hook, evicted after)
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        seeker_layer_prefix: str,    # prefix in the safetensors FILE  e.g. "text_model.encoder.layers"
        model_layer_prefix: str,     # prefix in the model OBJECT       e.g. "encoder.layers"
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        output_hidden_states: bool = True,
    ):
        self.model = model
        self.seeker = seeker
        self.seeker_layer_prefix = seeker_layer_prefix
        self.model_layer_prefix = model_layer_prefix
        self.device = device
        self.dtype = dtype
        self.output_hidden_states = output_hidden_states
        # Strip string to translate file keys -> model keys (e.g. "text_model." -> "")
        self._key_strip = seeker_layer_prefix[: len(seeker_layer_prefix) - len(model_layer_prefix)]

        self._install_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _get_layers(self):
        """Return the nn.ModuleList of encoder layers regardless of model hierarchy."""
        if hasattr(self.model, "text_model"):
            return self.model.text_model.encoder.layers
        elif hasattr(self.model, "encoder"):
            return self.model.encoder.layers
        else:
            raise AttributeError(f"Cannot find encoder layers on {type(self.model)}.")

    def _install_hooks(self):
        for i, layer in enumerate(self._get_layers()):
            layer._clip_layer_idx = i
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_hook(self, module: nn.Module, args):
        idx = module._clip_layer_idx

        # Keys as they appear in the safetensors FILE
        seeker_prefix = f"{self.seeker_layer_prefix}.{idx}."
        file_keys = [k for k in self.seeker.weight_map if k.startswith(seeker_prefix)]

        # Fetch raw tensors using original file keys
        raw_sd = self.seeker.get_tensors(file_keys, device=self.device, dtype=self.dtype)

        # Translate to model-side key names (strip key_strip prefix if needed)
        if self._key_strip:
            sd = {
                k[len(self._key_strip):] if k.startswith(self._key_strip) else k: v
                for k, v in raw_sd.items()
            }
        else:
            sd = raw_sd

        place_tensors(self.model, sd, self.device, self.dtype)

    def _post_hook(self, module: nn.Module, args, output):
        evict_module(module)
        return output

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_cls,
        model_dir: str,
        subfolder: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        output_hidden_states: bool = True,
        cache_to_ram: bool = False
    ) -> "CLIPTextModelStreamer":
        import os
        path = os.path.join(model_dir, subfolder)

        print(f"Initializing SafetensorsLiveSeeker on {subfolder} weights ...")
        seeker = get_seeker(path, cache_to_ram=cache_to_ram)

        print(f"Loading resident tensors to GPU for {subfolder} ...")

        # Instantiate on meta device
        config = model_cls.config_class.from_pretrained(path)
        with default_dtype(dtype), init_empty_weights():
            model = model_cls(config)
        model.eval()

        # Move any non-meta buffers (e.g. position_ids) to device immediately
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                place_tensors(model, {buf_name: buf}, device, buf.dtype if buf.is_floating_point() else torch.float32)

        # ------------------------------------------------------------------
        # Detect safetensors key structure vs. model object structure
        # ------------------------------------------------------------------
        # Safetensors may use "text_model.encoder.layers.*" prefix
        # Newer transformers CLIPTextModel is flattened: "encoder.layers.*"
        # CLIPTextModelWithProjection still uses "text_model.encoder.layers.*" on both sides
        sample_keys = list(seeker.weight_map.keys())[:10]
        file_has_text_model_prefix = any(k.startswith("text_model.") for k in sample_keys)

        # Check model object structure
        model_has_text_model = any(
            name == "text_model" or name.startswith("text_model.")
            for name, _ in model.named_modules()
        )

        # seeker_layer_prefix: what the FILE uses for encoder layers
        seeker_layer_prefix = "text_model.encoder.layers" if file_has_text_model_prefix else "encoder.layers"
        seeker_streaming_prefix = seeker_layer_prefix + "."

        # model_layer_prefix: what the MODEL OBJECT uses for encoder layers
        model_layer_prefix = "text_model.encoder.layers" if model_has_text_model else "encoder.layers"

        # ------------------------------------------------------------------
        # Load resident weights (everything except the streaming layers)
        # ------------------------------------------------------------------
        resident_keys = [k for k in seeker.weight_map if not k.startswith(seeker_streaming_prefix)]
        resident_sd_raw = seeker.get_tensors(resident_keys, device="cpu", dtype=dtype)

        # Translate file keys -> model keys for resident tensors
        key_strip = seeker_layer_prefix[: len(seeker_layer_prefix) - len(model_layer_prefix)]
        if key_strip:
            resident_sd = {
                k[len(key_strip):] if k.startswith(key_strip) else k: v
                for k, v in resident_sd_raw.items()
            }
        else:
            resident_sd = resident_sd_raw

        cpu_sd = {k: v for k, v in resident_sd.items() if "token_embedding" in k}
        gpu_sd = {k: v for k, v in resident_sd.items() if k not in cpu_sd}

        if cpu_sd:
            place_tensors(model, cpu_sd, "cpu", dtype)
            from weellm.memory import pin_module_to_cpu
            if model_has_text_model:
                pin_module_to_cpu(model, "text_model.embeddings.token_embedding")
            else:
                pin_module_to_cpu(model, "embeddings.token_embedding")

        if gpu_sd:
            place_tensors(model, gpu_sd, device, dtype)
            
        del resident_sd_raw, resident_sd, cpu_sd, gpu_sd
        clean_memory(device)

        _patch_forward_input_device(model, device)

        if model_has_text_model:
            num_layers = len(model.text_model.encoder.layers)
        else:
            num_layers = len(model.encoder.layers)
        print(f"  -> {num_layers} encoder layers will stream on-demand. Resident weights on GPU.")

        return cls(model, seeker, seeker_layer_prefix, model_layer_prefix, device, dtype, output_hidden_states)

    def __call__(self, *args, **kwargs):
        if self.output_hidden_states:
            kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = False
        return self.model(*args, **kwargs)
