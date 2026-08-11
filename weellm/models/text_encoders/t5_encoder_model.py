"""
t5_streamer.py -- Hook-based layer-streaming for T5EncoderModel.

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
from weellm.memory import place_tensors
from accelerate.utils.modeling import set_module_tensor_to_device



class T5EncoderModelStreamer:
    """
    Wraps T5EncoderModel for layer-by-layer streaming.

    Resident: shared.weight (embeddings), encoder.final_layer_norm
    Streamed:  encoder.block[i]  (loaded per-hook, evicted after)

    Note: encoder.block.0 contains the relative_attention_bias weights
    which are shared across all blocks. We keep these resident.
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
        # - encoder.block.0.layer.0.SelfAttention.relative_attention_bias  (shared bias, block 0 only)
        self._resident_key_prefixes = (
            "shared.",
            "encoder.final_layer_norm.",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
        )

        self._install_hooks()

    def _get_block_keys(self, block_idx: int):
        """Return all keys for block[block_idx], excluding always-resident keys."""
        prefix = f"encoder.block.{block_idx}."
        keys = [k for k in self.seeker.weight_map if k.startswith(prefix)]
        # Exclude relative_attention_bias from block 0 streaming (it's resident)
        keys = [k for k in keys if "relative_attention_bias" not in k]
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

    def _evict_block(self, module: nn.Module, block_idx: int) -> None:
        """
        Evict a T5 encoder block back to meta device.

        Block 0 is special: it owns ``relative_attention_bias`` which T5 uses
        to compute ``position_bias`` and then *passes that tensor forward* to
        every subsequent block.  We must keep ``relative_attention_bias`` resident
        on GPU for the entire encode pass, otherwise the forward through blocks
        1..N-1 will see a CUDA ``position_bias`` but try to touch meta-device
        tensors inside block 0's attention sub-module, causing:
            RuntimeError: Tensor on device cuda:0 is not on the expected device meta!

        Solution: when evicting block 0, skip the relative_attention_bias sub-module.
        For all other blocks evict everything as usual.
        """
        from weellm.memory import evict_module as _evict
        from accelerate.utils.modeling import set_module_tensor_to_device

        if block_idx != 0:
            _evict(module)
            return

        # Block 0: evict everything EXCEPT relative_attention_bias
        for name, param in list(module.named_parameters(recurse=True)):
            if "relative_attention_bias" in name:
                continue  # keep resident
            dev = getattr(param, "device", None)
            if dev is not None and dev.type != "meta":
                try:
                    set_module_tensor_to_device(module, name, "meta")
                except Exception:
                    pass

        for name, buf in list(module.named_buffers(recurse=True)):
            if "relative_attention_bias" in name:
                continue
            dev = getattr(buf, "device", None)
            if dev is not None and dev.type != "meta":
                try:
                    set_module_tensor_to_device(module, name, "meta")
                except Exception:
                    pass

    def _post_hook(self, module: nn.Module, args, output):
        self._evict_block(module, module._t5_block_idx)
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
        cache_to_ram: bool = False
    ) -> "T5EncoderModelStreamer":
        from transformers import T5EncoderModel

        print(f"Initializing SafetensorsLiveSeeker on text_encoder_2 weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        print(f"  Found {len(seeker.weight_map)} tensors.")

        print(f"Instantiating T5EncoderModel on meta device ...")
        config = T5EncoderModel.config_class.from_pretrained(model_dir)
        with default_dtype(dtype), init_empty_weights():
            model = T5EncoderModel(config)
        model.eval()

        # Move any non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        # Identify resident keys
        resident_prefixes = (
            "shared.",
            "encoder.final_layer_norm.",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
        )

        def is_resident(k):
            return any(k.startswith(p) or k == p for p in resident_prefixes)

        resident_keys = [k for k in seeker.weight_map if is_resident(k)]
        print(f"Loading resident T5 tensors to GPU ({len(resident_keys)} tensors) ...")
        resident_sd = seeker.get_tensors(resident_keys, device=device,
            dtype=dtype)
        place_tensors(model, resident_sd, device, dtype)
        
        # T5 ties encoder.embed_tokens.weight to shared.weight, but loading via accelerate 
        # breaks the pointer when weights are on meta device. Manually tie them back.
        if hasattr(model.encoder, "embed_tokens") and hasattr(model, "shared"):
            model.encoder.embed_tokens.weight = model.shared.weight
            
        del resident_sd
        clean_memory(device)

        num_blocks = len(model.encoder.block)
        print(f"  -> {num_blocks} T5 encoder blocks will stream on-demand. Resident weights on GPU.")
        report_memory("After T5 encoder init")

        return cls(model, seeker, device, dtype, max_length)

    def __call__(self, *args, **kwargs):
        kwargs["return_dict"] = False
        return self.model(*args, **kwargs)
