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

from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

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

        _apply_state_dict(self.model, sd, self.device, self.dtype)
        module._clip_loaded_params = list(sd.keys())  # model-side names for eviction

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_clip_loaded_params", []))
        module._clip_loaded_params = []
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
        with init_empty_weights():
            model = model_cls(config)
        model.eval()

        # Move any non-meta buffers (e.g. position_ids) to device immediately
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

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
        resident_sd_raw = seeker.get_tensors(resident_keys, device=device,
            dtype=dtype)

        # Translate file keys -> model keys for resident tensors
        key_strip = seeker_layer_prefix[: len(seeker_layer_prefix) - len(model_layer_prefix)]
        if key_strip:
            resident_sd = {
                k[len(key_strip):] if k.startswith(key_strip) else k: v
                for k, v in resident_sd_raw.items()
            }
        else:
            resident_sd = resident_sd_raw

        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd_raw, resident_sd
        clean_memory(device)

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
