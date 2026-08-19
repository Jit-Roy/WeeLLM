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

logger = logging.getLogger("weellm")

# Prefixes for blocks that are streamed (not resident).
_DECODER_STREAMING_PREFIXES = ("decoder.transformer_blocks.",)
# Encoder keys — kept on meta until encode() is first called.
_ENCODER_PREFIX = "encoder."


class MiniMaxVAEStreamer:
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
        self.model  = model
        self.seeker = seeker
        self.device = device
        self.dtype  = dtype
        # Real latents_mean/std/latent_channels from config.json
        self._config_extras = config_extras or {}
        # Call counter for progress logging (one full pass = 36 blocks × N temporal chunks)
        self._call_counter: int = 0

        self._encoder_loaded = False
        self._encoder_lock   = threading.Lock()

        self._install_decoder_hooks()
        self._patch_encode()

    def _get_resident_keys(self) -> List[str]:
        return [
            k for k in self.seeker.weight_map
            if not any(k.startswith(p) for p in _DECODER_STREAMING_PREFIXES)
            and not k.startswith(_ENCODER_PREFIX)
        ]

    def _get_encoder_keys(self) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(_ENCODER_PREFIX)]

    def _get_block_keys(self, shard_prefix: str) -> List[str]:
        return [k for k in self.seeker.weight_map if k.startswith(shard_prefix + ".")]

    def _place_tensor(self, name: str, tensor: torch.Tensor, target_device: str, target_dtype: torch.dtype) -> None:
        if "num_batches_tracked" in name:
            return
        
        # map safetensors key to MiniMaxH3VideoVAE's inner module
        mapped_name = "model." + name

        if tensor.is_floating_point():
            set_module_tensor_to_device(self.model, mapped_name, target_device, value=tensor, dtype=target_dtype)
        else:
            set_module_tensor_to_device(self.model, mapped_name, target_device, value=tensor)

    def _evict_keys(self, keys: List[str]) -> None:
        for name in keys:
            mapped_name = "model." + name
            try:
                set_module_tensor_to_device(self.model, mapped_name, "meta")
            except Exception:
                pass

    def _install_decoder_hooks(self) -> None:
        decoder = getattr(self.model.model, "decoder", None)
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
        return output

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

            kwargs.pop("return_dict", None)
            
            # Cast first positional arg to correct dtype if it's a tensor
            if args and isinstance(args[0], torch.Tensor):
                args = (args[0].to(self.dtype),) + args[1:]
                
            result = original_encode(*args, **kwargs)
            
            # Wrap in diffusers' DiagonalGaussianDistribution
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            posterior = DiagonalGaussianDistribution(result)

            with self._encoder_lock:
                if self._encoder_loaded:
                    logger.info("[WeeLLM VAE] Encoder forward complete — evicting encoder weights ...")
                    self._evict_keys(self._get_encoder_keys())
                    clean_memory(self.device)
                    report_memory("After VAE Encoder Eviction")
                    self._encoder_loaded = False

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
        return 5

    @property
    def config(self):
        original_config = getattr(self.model, "config", {})
        
        class ConfigProxy:
            def __init__(self, c, extra):
                self._c = c
                self._extra = extra
            def __getattr__(self, name):
                # Prefer extra dict (from config.json) for specific known fields
                if name in self._extra:
                    return self._extra[name]
                return getattr(self._c, name)
                
        return ConfigProxy(original_config, self._config_extras)

    def decode(self, latents, return_dict=True, **kwargs):
        """Wrap inner VAE decode via decode_base() which routes through decode_temporal()
        for proper temporal chunking (tokens_chunk_size=5 latent frames at a time).
        The raw AutoencoderKLLegacy.decode() has no chunking → OOM on 4GB VRAM."""
        import os
        import torch
        
        cache_file = r"D:\Personal Projects\LightLLM\.weellm_cache\vae_decode_cache.pt"
        if os.path.exists(cache_file):
            print(f"    [VAE Streamer] Loading cached decoded video from {cache_file} ...", flush=True)
            result = torch.load(cache_file, map_location=latents.device)
            print(f"    [VAE Streamer] Cached decode loaded: {tuple(result.shape)}", flush=True)
        else:
            # self.model is MiniMaxH3VideoVAE; self.model.model is AutoencoderKLLegacy
            inner = self.model.model
            with torch.no_grad():
                report_memory("Before VAE decode_base")
                # decode_base → decode_temporal → chunks of tokens_chunk_size latent frames
                # Note: decode_base internally routes to decode() which applies post_quant_conv
                result = inner.decode_base(latents)
                report_memory("After VAE decode_base")
                print(f"    [VAE Streamer] decode_base done: {tuple(result.shape)}", flush=True)
                
            print(f"    [VAE Streamer] Saving video decode cache to {cache_file} ...", flush=True)
            torch.save(result.cpu(), cache_file)
            result = result.to(latents.device)


        if return_dict:
            return result
        # Diffusers decode blocks do: vae.decode(..., return_dict=False)[0]
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
    ) -> "MiniMaxVAEStreamer":
        vae_dir = Path(vae_dir)
        config_path = vae_dir / "config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on MiniMax VAE weights ...")
        source_path = vae_dir / config["source_path"]
        seeker = get_seeker(source_path, cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating MiniMaxH3VideoVAE on meta device ...")
        
        import importlib.util
        import importlib.machinery
        import importlib
        import sys
        
        # Load custom class by creating a synthetic package to avoid relative import errors
        pkg_name = "minimax_vae_pkg"
        if pkg_name not in sys.modules:
            spec = importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
            pkg = importlib.util.module_from_spec(spec)
            pkg.__path__ = [str(vae_dir)]
            sys.modules[pkg_name] = pkg

        minimax_h3_video_vae = importlib.import_module(f"{pkg_name}.minimax_h3_video_vae")
        klvae = importlib.import_module(f"{pkg_name}.klvae")

        MiniMaxH3VideoVAE = minimax_h3_video_vae.MiniMaxH3VideoVAE
        AutoencoderKLLegacy = klvae.AutoencoderKLLegacy

        load_kwargs = {
            "clip_length": int(config["vae_clip_length"]),
            "token_drop": int(config["vae_token_drop"]),
            "encoder_tiling": int(config["vae_encoder_tiling"]),
            "decoder_tiling": int(config["vae_decoder_tiling"]),
            "parallel_tiling": int(config["vae_parallel_tiling"]),
            # We must use spatial tiling (tile_size=256) to ensure the 16x16 blocks get cross-block 
            # self-attention smoothing. Disabling this (99999) causes visible grid artifacts in the final video.
            "tile_size": int(config["vae_tile_size"]) if "vae_tile_size" in config else 256,
            "tile_overlap_min": int(config["vae_tile_overlap_min"]),
            "encoder_parallel": int(config["vae_encoder_parallel"]),
            "decoder_parallel": int(config["vae_decoder_parallel"]),
            "chunk_dim": int(config["vae_chunk_dim"]),
        }

        with default_dtype(dtype), init_empty_weights():
            source_config = AutoencoderKLLegacy.load_config(str(source_path))
            inner_model, _ = AutoencoderKLLegacy.from_config(
                source_config, return_unused_kwargs=True, **load_kwargs
            )
            model = MiniMaxH3VideoVAE(inner_model)
        model.eval()

        # Initialise the parallel state dict that tiled_decode() reads sp_rank/sp_size from.
        # MiniMaxH3VideoVAE.from_pretrained() does this, but our custom path skips it.
        if bool(config.get("vae_parallel_tiling", 0)):
            minimax_h3_video_vae._ensure_vae_parallel_state()
        else:
            # Always seed the state for single-process inference to be safe.
            _parallel = minimax_h3_video_vae.get_parallel_state()
            if not _parallel:
                _parallel.update({
                    "group_size": 1, "group_rank": 0, "local_process_group": None,
                    "sp_size": 1, "sp_rank": 0, "sp_enabled": False, "sp_process_group": None,
                    "tp_size": 1, "tp_rank": 0,
                })

        # Extract real latent normalisation stats to pass into config proxy
        config_extras = {
            "latent_channels": config.get("latent_channels", 24),
            "latents_mean": config.get("latents_mean", [0.0] * 24),
            "latents_std":  config.get("latents_std",  [1.0] * 24),
            "clip_length":  config.get("vae_clip_length", 17),
        }
        streamer = cls(model, seeker, device, dtype, config_extras=config_extras)

        logger.info("  Step 3/3 -- Loading VAE resident tensors to GPU ...")
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for name, tensor in resident_sd.items():
            streamer._place_tensor(name, tensor, device, dtype)
        del resident_sd

        clean_memory(device)
        report_memory("After VAE resident load")
        logger.info("  MiniMaxVAEStreamer ready (%d resident keys, decoder blocks streamed).", len(resident_keys))
        return streamer
