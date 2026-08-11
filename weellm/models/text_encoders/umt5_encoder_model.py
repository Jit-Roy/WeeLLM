"""
t5_streamer.py -- Hook-based layer-streaming for UMT5EncoderModel.

Strategy:
  - Resident on GPU: shared.weight (token embeddings), encoder.final_layer_norm
  - Streamed:        encoder.block[0..N]  (loaded just-in-time, evicted after)

T5 encoder block structure in safetensors:
  encoder.block.{i}.layer.0.SelfAttention.*
  encoder.block.{i}.layer.0.layer_norm.*
  encoder.block.{i}.layer.1.DenseReluDense.*
  encoder.block.{i}.layer.1.layer_norm.*

Shared bias tensors live at:
  encoder.block.0.layer.0.SelfAttention.relative_attention_bias.*
  (only in block 0, referenced by all blocks)

Uses the same accelerate-based set_module_tensor_to_device pattern.
"""

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from weellm.utils import default_dtype
from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory
from weellm.memory import place_tensors, evict_module
from accelerate.utils.modeling import set_module_tensor_to_device



class UMT5EncoderModelStreamer:
    """
    Wraps UMT5EncoderModel for layer-by-layer streaming.

    Resident: shared.weight (embeddings), encoder.final_layer_norm
    Streamed:  encoder.block[i]  (loaded per-hook, evicted after)
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
    ):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self._streaming_block_prefix = "encoder.block."
        # Keys that are resident (not streamed per-block):
        # - shared.weight
        # - encoder.final_layer_norm
        self._resident_key_prefixes = (
            "shared.",
            "encoder.final_layer_norm.",
        )

        self._install_hooks()

    def _get_block_keys(self, block_idx: int):
        """Return all keys for block[block_idx], excluding always-resident keys."""
        prefix = f"encoder.block.{block_idx}."
        keys = [k for k in self.seeker.weight_map if k.startswith(prefix)]
        return keys

    def _install_hooks(self):
        for i, block in enumerate(self.model.encoder.block):
            block._t5_block_idx = i
            block.register_forward_pre_hook(self._pre_hook)
            block.register_forward_hook(self._post_hook)

    def _pre_hook(self, module: nn.Module, args):
        idx = module._t5_block_idx
        block_keys = self._get_block_keys(idx)
        sd = self.seeker.get_tensors(block_keys, device=self.device, dtype=self.dtype)
        place_tensors(self.model, sd, self.device, self.dtype)

    def _post_hook(self, module: nn.Module, args, output):
        evict_module(module)
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
        cache_to_ram: bool = False
    ) -> "UMT5EncoderModelStreamer":
        from transformers import UMT5EncoderModel

        print(f"Initializing SafetensorsLiveSeeker on text_encoder weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors.")

        print(f"Instantiating UMT5EncoderModel on meta device ...")
        config = UMT5EncoderModel.config_class.from_pretrained(model_dir)
        with default_dtype(dtype), init_empty_weights():
            model = UMT5EncoderModel(config)
        model.eval()

        # Move any non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        # Identify resident keys
        resident_prefixes = (
            "shared.",
            "encoder.final_layer_norm.",
        )

        def is_resident(k):
            return any(k.startswith(p) or k == p for p in resident_prefixes)

        resident_keys = [k for k in seeker.weight_map if is_resident(k)]
        print(f"Loading resident UMT5 tensors ({len(resident_keys)} tensors) ...")
        resident_sd = seeker.get_tensors(resident_keys, device="cpu", dtype=dtype)
        
        cpu_sd = {k: v for k, v in resident_sd.items() if k.startswith("shared.")}
        gpu_sd = {k: v for k, v in resident_sd.items() if k not in cpu_sd}
        
        if cpu_sd:
            place_tensors(model, cpu_sd, "cpu", dtype)
            from weellm.memory import pin_module_to_cpu
            pin_module_to_cpu(model, "shared")
            if hasattr(model.encoder, "embed_tokens"):
                pin_module_to_cpu(model, "encoder.embed_tokens")
                
        if gpu_sd:
            place_tensors(model, gpu_sd, device, dtype)
        
        # UMT5 ties encoder.embed_tokens.weight to shared.weight, but loading via accelerate 
        # breaks the pointer when weights are on meta device. Manually tie them back.
        if hasattr(model.encoder, "embed_tokens") and hasattr(model, "shared"):
            model.encoder.embed_tokens.weight = model.shared.weight
            
        del resident_sd, cpu_sd, gpu_sd
        clean_memory(device)

        num_blocks = len(model.encoder.block)
        print(f"  -> {num_blocks} UMT5 encoder blocks will stream on-demand. Resident weights on GPU.")
        report_memory("After UMT5 encoder init")

        return cls(model, seeker, device, dtype, max_length)

    def __call__(self, *args, **kwargs):
        kwargs["return_dict"] = False
        return self.model(*args, **kwargs)
