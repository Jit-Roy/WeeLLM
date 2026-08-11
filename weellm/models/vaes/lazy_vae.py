"""
lazy_vae.py -- Memory-efficient VAE streamer for WeeLLM.

Strategy
--------
- **Resident** on GPU at all times: quant_conv, post_quant_conv,
  decoder.conv_in, decoder.conv_norm_out, decoder.conv_out
  (all small; always needed at the start/end of a decode pass).
- **Streamed** via forward hooks: decoder.mid_block, decoder.up_blocks[i]
  (loaded just-in-time, evicted immediately after each block's forward pass).
- **Lazy encoder**: encoder weights stay on the meta device until the first
  call to encode() (required for image-to-image). They are loaded, used,
  then evicted back to meta so VRAM is freed after encoding.
"""

import importlib
import json
import logging
import threading
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from weellm.utils import default_dtype
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.seeker import get_seeker
from weellm.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")

# Prefixes for blocks that are streamed (not resident).
_DECODER_STREAMING_PREFIXES = ("decoder.mid_block.", "decoder.up_blocks.")
# Encoder keys — kept on meta until encode() is first called.
_ENCODER_PREFIX = "encoder."


# ---------------------------------------------------------------------------
# Key-mapping helpers (legacy safetensors attention key names → diffusers)
# ---------------------------------------------------------------------------

def _map_vae_key(name: str) -> str:
    mapped = name
    if ".query." in mapped:
        mapped = mapped.replace(".query.", ".to_q.")
    if ".key." in mapped:
        mapped = mapped.replace(".key.", ".to_k.")
    if ".value." in mapped:
        mapped = mapped.replace(".value.", ".to_v.")
    if ".proj_attn." in mapped:
        mapped = mapped.replace(".proj_attn.", ".to_out.0.")
    return mapped


def _has_module_path(model: nn.Module, name: str) -> bool:
    current = model
    for part in name.split(".")[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
        if current is None:
            return False
    return hasattr(current, name.split(".")[-1])


def _resolve_vae_key(model: nn.Module, name: str) -> str:
    mapped = _map_vae_key(name)
    if _has_module_path(model, mapped):
        return mapped
    if _has_module_path(model, name):
        return name
    return mapped


# ---------------------------------------------------------------------------
# LazyVAEStreamer
# ---------------------------------------------------------------------------

class LazyVAEStreamer:
    """
    Memory-efficient VAE wrapper using a resident/streaming split.

    Decoder large blocks (mid_block, up_blocks) are streamed layer-by-layer
    via forward hooks.  Small connector layers remain resident.  The encoder
    is lazy-loaded on first use (needed for image-to-image) and evicted
    afterwards.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.model  = model
        self.seeker = seeker
        self.device = device
        self.dtype  = dtype

        self._encoder_loaded = False
        self._encoder_lock   = threading.Lock()

        self._install_decoder_hooks()
        self._patch_encode()

    # ------------------------------------------------------------------
    # Resident / streaming key partitioning
    # ------------------------------------------------------------------

    def _get_resident_keys(self) -> List[str]:
        """Keys loaded once at startup and kept on GPU permanently."""
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _DECODER_STREAMING_PREFIXES)
            and not k.startswith(_ENCODER_PREFIX)
        ]

    def _get_encoder_keys(self) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(_ENCODER_PREFIX)]

    def _get_block_keys(self, shard_prefix: str) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(shard_prefix + ".")]

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _place_tensor(self, name: str, tensor: torch.Tensor, target_device: str, target_dtype: torch.dtype) -> None:
        mapped = _resolve_vae_key(self.model, name)
        if "num_batches_tracked" in name:
            return
        if tensor.is_floating_point():
            set_module_tensor_to_device(self.model, mapped, target_device, value=tensor, dtype=target_dtype)
        else:
            set_module_tensor_to_device(self.model, mapped, target_device, value=tensor)

    def _evict_keys(self, keys: List[str]) -> None:
        for name in keys:
            mapped = _resolve_vae_key(self.model, name)
            try:
                set_module_tensor_to_device(self.model, mapped, "meta")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Decoder streaming hooks
    # ------------------------------------------------------------------

    def _install_decoder_hooks(self) -> None:
        decoder = self.model.decoder
        if not hasattr(decoder, "mid_block") or not hasattr(decoder, "up_blocks"):
            logger.warning("[WeeLLM VAE] Decoder does not have expected mid_block/up_blocks — skipping streaming hooks.")
            return

        streaming_blocks = []
        if decoder.mid_block is not None:
            streaming_blocks.append(("decoder.mid_block", decoder.mid_block))
        for i, block in enumerate(decoder.up_blocks):
            streaming_blocks.append((f"decoder.up_blocks.{i}", block))

        for shard_prefix, block in streaming_blocks:
            block._vae_shard_prefix  = shard_prefix
            block._vae_loaded_keys   = []
            block.register_forward_pre_hook(self._block_pre_hook)
            block.register_forward_hook(self._block_post_hook)

        logger.info(
            "      -> [WeeLLM VAE] Installed streaming hooks on %d decoder blocks (%s).",
            len(streaming_blocks),
            ", ".join(p for p, _ in streaming_blocks),
        )

    def _block_pre_hook(self, module: nn.Module, args):
        shard_prefix = module._vae_shard_prefix
        keys = self._get_block_keys(shard_prefix)
        sd   = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
        for name, tensor in sd.items():
            self._place_tensor(name, tensor, self.device, self.dtype)
        module._vae_loaded_keys = keys
        return args

    def _block_post_hook(self, module: nn.Module, args, output):
        self._evict_keys(getattr(module, "_vae_loaded_keys", []))
        module._vae_loaded_keys = []
        return output

    # ------------------------------------------------------------------
    # Lazy encoder (for image-to-image)
    # ------------------------------------------------------------------

    def _patch_encode(self) -> None:
        """Wrap model.encode() to lazy-load encoder weights on first call."""
        original_encode = self.model.encode

        def _lazy_encode(self_obj, *args, **kwargs):
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

            result = original_encode(*args, **kwargs)

            # Evict encoder weights back to meta to reclaim VRAM.
            with self._encoder_lock:
                if self._encoder_loaded:
                    logger.info("[WeeLLM VAE] Encoder forward complete — evicting encoder weights ...")
                    self._evict_keys(self._get_encoder_keys())
                    clean_memory(self.device)
                    report_memory("After VAE Encoder Eviction")
                    self._encoder_loaded = False

            return result

        self.model.encode = _lazy_encode.__get__(self.model, self.model.__class__)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        vae_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
    ) -> "LazyVAEStreamer":
        vae_dir     = Path(vae_dir)
        config_path = vae_dir / "config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        class_name = cfg_dict.get("_class_name", "AutoencoderKL")
        diffusers  = importlib.import_module("diffusers")
        vae_cls    = getattr(diffusers, class_name)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on VAE weights (%s) ...", class_name)
        seeker = get_seeker(vae_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating %s on meta device ...", class_name)
        with default_dtype(dtype), init_empty_weights():
            cfg   = vae_cls.load_config(str(config_path))
            model = vae_cls.from_config(cfg)
        model.eval()

        streamer = cls(model, seeker, device, dtype)

        logger.info("  Step 3/3 -- Loading VAE resident tensors to GPU ...")
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for name, tensor in resident_sd.items():
            streamer._place_tensor(name, tensor, device, dtype)
        del resident_sd

        # Flux2-family VAEs have a BatchNorm layer accessed BEFORE decode() is called.
        # Load bn buffers eagerly on CPU so they are not meta tensors.
        if hasattr(model, "bn"):
            logger.info("  Eagerly loading VAE BN layers (CPU) to preserve contrast ...")
            bn_keys = [k for k in seeker.weight_map if "bn." in k]
            bn_sd   = seeker.get_tensors(bn_keys, device="cpu", dtype=torch.float32)
            for name, tensor in bn_sd.items():
                streamer._place_tensor(name, tensor, "cpu", torch.float32)
            del bn_sd

        clean_memory(device)
        report_memory("After VAE resident load")
        logger.info("  LazyVAEStreamer ready (%d resident keys, decoder blocks streamed).", len(resident_keys))
        return streamer
