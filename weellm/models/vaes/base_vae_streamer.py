"""
base_vae_streamer.py -- Shared base class for VAE streamers.
"""

import logging
import threading
from typing import List

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

_ENCODER_PREFIX = "encoder."


class BaseVAEStreamer:
    """Base class for VAE streamers."""
    
    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        
        self._encoder_loaded = False
        self._encoder_lock = threading.Lock()
        
    @property
    def decoder_streaming_prefixes(self) -> tuple:
        """Override in subclasses to provide prefixes for streamed decoder blocks."""
        return ()
        
    def _get_resident_keys(self) -> List[str]:
        """Keys loaded once at startup and kept on GPU permanently."""
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in self.decoder_streaming_prefixes)
            and not k.startswith(_ENCODER_PREFIX)
        ]

    def _get_encoder_keys(self) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(_ENCODER_PREFIX)]

    def _get_block_keys(self, shard_prefix: str) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(shard_prefix + ".")]

    def _resolve_vae_key(self, name: str) -> str:
        """Override in subclasses to map keys (e.g. legacy diffusers keys)."""
        return name

    def _place_tensor(self, name: str, tensor: torch.Tensor, target_device: str, target_dtype: torch.dtype) -> None:
        mapped = self._resolve_vae_key(name)
        if "num_batches_tracked" in mapped:
            return
        if tensor.is_floating_point():
            set_module_tensor_to_device(self.model, mapped, target_device, value=tensor, dtype=target_dtype)
        else:
            set_module_tensor_to_device(self.model, mapped, target_device, value=tensor)

    def _evict_keys(self, keys: List[str]) -> None:
        for name in keys:
            mapped = self._resolve_vae_key(name)
            try:
                set_module_tensor_to_device(self.model, mapped, "meta")
            except Exception:
                pass

    def _load_encoder(self):
        with self._encoder_lock:
            if not self._encoder_loaded:
                logger.info("\n[WeeLLM VAE] Lazy encoder triggered — loading encoder weights to GPU ...")
                enc_keys = self._get_encoder_keys()
                enc_sd   = self.seeker.get_tensors(enc_keys, device=self.device, dtype=self.dtype)
                for name, tensor in enc_sd.items():
                    self._place_tensor(name, tensor, self.device, self.dtype)
                del enc_sd
                report_memory("After VAE Encoder Load")
                self._encoder_loaded = True

    def _evict_encoder(self):
        with self._encoder_lock:
            if self._encoder_loaded:
                logger.info("[WeeLLM VAE] Encoder forward complete — evicting encoder weights ...")
                self._evict_keys(self._get_encoder_keys())
                clean_memory(self.device)
                report_memory("After VAE Encoder Eviction")
                self._encoder_loaded = False
