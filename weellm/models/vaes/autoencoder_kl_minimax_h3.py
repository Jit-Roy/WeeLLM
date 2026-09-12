import json
import logging
import threading
from pathlib import Path
from typing import List, Union
import sys

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.seeker import get_seeker
from weellm.utils import default_dtype, clean_memory, report_memory
from .base_vae_streamer import BaseVAEStreamer

logger = logging.getLogger("weellm")

class AutoencoderKLMiniMaxH3Streamer(BaseVAEStreamer):
    """
    Memory-efficient VAE wrapper for the 10GB MiniMax-H3 Video VAE.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str,
        dtype: torch.dtype,
        config_extras: dict = None,
    ) -> None:
        super().__init__(model, seeker, device, dtype)
        # Real latents_mean/std/latent_channels from config.json
        self._config_extras = config_extras or {}
        # Call counter for progress logging (one full pass = 36 blocks × N temporal chunks)
        self._call_counter: int = 0

        self._install_decoder_hooks()
        self._patch_encode()

    @property
    def decoder_streaming_prefixes(self) -> tuple:
        return ("decoder.transformer_blocks.",)

    def _install_decoder_hooks(self) -> None:
        # self.model is AutoencoderKLMiniMaxH3; decoder is a direct attribute
        decoder = getattr(self.model, "decoder", None)
        if decoder is None or not hasattr(decoder, "transformer_blocks"):
            logger.warning("[WeeLLM VAE] Decoder does not have expected transformer_blocks — skipping hooks.")
            return

        streaming_blocks = []
        for i, block in enumerate(decoder.transformer_blocks):
            streaming_blocks.append((f"decoder.transformer_blocks.{i}", block))

        for shard_prefix, block in streaming_blocks:
            block._vae_shard_prefix  = shard_prefix
            block._vae_loaded_keys   = []
            block.register_forward_pre_hook(self._block_pre_hook)
            block.register_forward_hook(self._block_post_hook)

        logger.info(
            "      -> [WeeLLM VAE] Installed streaming hooks on %d MiniMax VAE decoder blocks.",
            len(streaming_blocks)
        )

    def _block_pre_hook(self, module: nn.Module, args):
        shard_prefix = module._vae_shard_prefix
        keys = self._get_block_keys(shard_prefix)
        # Load block from disk → GPU (no spatial tiling → only 8 temporal calls per block)
        sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
        for name, tensor in sd.items():
            self._place_tensor(name, tensor, self.device, self.dtype)
        module._vae_loaded_keys = keys

        # Progress: print every 36 calls = one temporal chunk fully decoded
        self._call_counter += 1
        if self._call_counter % 36 == 0:
            chunk_num = self._call_counter // 36
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                print(f"    [VAE Streamer] Temporal chunk #{chunk_num} done  VRAM {used:.2f}/{resv:.2f} GB", flush=True)
            else:
                print(f"    [VAE Streamer] Temporal chunk #{chunk_num} done", flush=True)

        return args

    def _block_post_hook(self, module: nn.Module, args, output):
        # Evict block weights from GPU — CPU has no cache so this is final
        self._evict_keys(getattr(module, "_vae_loaded_keys", []))
        module._vae_loaded_keys = []
        # Aggressively clear fragmented VRAM every 2 blocks instead of 36
        if self._call_counter % 2 == 0:
            torch.cuda.empty_cache()
            if hasattr(torch, "clear_autocast_cache"):
                torch.clear_autocast_cache()
        return output

    def _patch_encode(self) -> None:
        """Wrap model.encode() to lazy-load encoder weights on first call."""
        original_encode = self.model.encode

        def _lazy_encode(self_obj, *args, **kwargs):
            self._load_encoder()

            kwargs.pop("return_dict", None)

            # Cast first positional arg to correct dtype if it's a tensor
            if args and isinstance(args[0], torch.Tensor):
                args = (args[0].to(self.dtype),) + args[1:]

            # AutoencoderKLMiniMaxH3.encode returns AutoencoderKLOutput with .latent_dist
            result = original_encode(*args, return_dict=True, **kwargs)
            if hasattr(result, "latent_dist"):
                posterior = result.latent_dist
            else:
                from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
                posterior = DiagonalGaussianDistribution(result)

            self._evict_encoder()

            return (posterior, )

        self.model.encode = _lazy_encode.__get__(self.model, self.model.__class__)

    @property
    def spatial_compression_ratio(self):
        return 16

    @property
    def temporal_compression_ratio(self):
        return 1
        
    @property
    def tokens_chunk_size(self):
        return 2


    @property
    def config(self):
        # AutoencoderKLMiniMaxH3 already has latents_mean, latents_std, latent_channels
        # on its config — return it directly with the config_extras merged in.
        original_config = getattr(self.model, "config", {})

        class ConfigProxy:
            def __init__(self, c, extra):
                self._c = c
                self._extra = extra
            def __getattr__(self, name):
                if name in self._extra:
                    return self._extra[name]
                return getattr(self._c, name)

        return ConfigProxy(original_config, self._config_extras)

    def decode(self, latents, return_dict=True, **kwargs):
        """Decode latents through the streaming VAE decoder.
        
        AutoencoderKLMiniMaxH3.decode() routes through self.decoder which is
        MiniMaxH3VideoViTDecoder3d with 36 transformer_blocks — our streaming
        hooks fire per-block to keep VRAM usage low.
        """
        import os
        cache_dir = os.path.join(os.getcwd(), ".weellm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "vae_decode_cache.pt")
        if os.path.exists(cache_file):
            print(f"    [VAE Streamer] Loading cached decoded video from {cache_file} ...", flush=True)
            result = torch.load(cache_file, map_location=latents.device)
            print(f"    [VAE Streamer] Cached decode loaded: {tuple(result.shape)}", flush=True)
        else:
            with torch.no_grad():
                report_memory("Before VAE decode")
                if hasattr(self.model, "tokens_chunk_size"):
                    self.model.tokens_chunk_size = 2
                out = self.model.decode(latents.to(self.dtype), return_dict=False)

                # out is a tuple; out[0] is the decoded video tensor
                result = out[0] if isinstance(out, (tuple, list)) else out
                report_memory("After VAE decode")
                print(f"    [VAE Streamer] decode done: {tuple(result.shape)}", flush=True)

            print(f"    [VAE Streamer] Saving video decode cache to {cache_file} ...", flush=True)
            torch.save(result.cpu(), cache_file)
            result = result.to(latents.device)

        if return_dict:
            return result
        if isinstance(result, torch.Tensor):
            return (result,)
        if isinstance(result, (tuple, list)):
            return result
        if hasattr(result, "sample"):
            return (result.sample,)
        return (result,)

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    @classmethod
    def from_pretrained(
        cls,
        vae_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
    ) -> "AutoencoderKLMiniMaxH3Streamer":
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
        from accelerate import init_empty_weights

        vae_dir = Path(vae_dir)
        config_path = vae_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on MiniMax VAE weights ...")
        # The VAE weights are safetensors shards in the same dir as config.json.
        # Older versions stored a custom 'source_path' key; the standard diffusers
        # config does not have it — fall back to vae_dir itself.
        source_path = vae_dir / config["source_path"] if "source_path" in config else vae_dir
        seeker = get_seeker(str(source_path), cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating AutoencoderKLMiniMaxH3 on meta device ...")
        # Strip diffusers metadata keys before passing to constructor
        init_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
        with init_empty_weights():
            model = AutoencoderKLMiniMaxH3(**init_cfg)
        model.eval()

        # Extract latent normalisation stats for the config proxy
        config_extras = {
            "latent_channels": config.get("latent_channels", 24),
            "latents_mean":    config.get("latents_mean", [0.0] * 24),
            "latents_std":     config.get("latents_std",  [1.0] * 24),
            "clip_length":     config.get("clip_length",  17),
        }
        streamer = cls(model, seeker, device, dtype, config_extras=config_extras)

        logger.info("  Step 3/3 -- Loading VAE resident tensors to GPU ...")
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for name, tensor in resident_sd.items():
            streamer._place_tensor(name, tensor, device, dtype)
        del resident_sd

        # Non-persistent buffers (e.g. decoder.rope.inv_freq) are computed in
        # __init__ under init_empty_weights() and land on CPU — they are NOT in
        # the weight_map, so resident loading skips them.  Move any CPU buffers
        # in the decoder to the target device now so the RoPE forward doesn't
        # crash with a cuda:0 vs cpu device mismatch.
        _decoder = getattr(model, "decoder", None)
        if _decoder is not None:
            for buf_name, buf in list(_decoder.named_buffers()):
                if buf.device.type == "cpu":
                    buf.data = buf.data.to(device)
                    logger.info("    [VAE] Moved decoder buffer '%s' → %s", buf_name, device)

        clean_memory(device)
        report_memory("After VAE resident load")
        logger.info("  MiniMaxVAEStreamer ready (%d resident keys, decoder blocks streamed).", len(resident_keys))
        return streamer
