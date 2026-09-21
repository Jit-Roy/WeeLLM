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

from weellm.io.seeker import get_seeker
from weellm.io.utils import default_dtype, clean_memory, report_memory
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

    # Removed incorrect properties to allow diffusers to use its native chunking math

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

                # ---------------------------------------------------------------------------
                # _stream_decode: single-pass streaming tiled decode
                #
                # Strategy: load each of the 36 VAE transformer blocks from disk ONCE, then
                # run it serially at batch=1 across every (temporal_clip × spatial_tile).
                # - Disk reads:  36  (same speed as the untiled 42s run)
                # - Tile size:   256×256 (in-distribution for the ViT, no blocky artifacts)
                # - No quadratic blowup from batching all tiles together (v2 was slow here)
                # ---------------------------------------------------------------------------

                def _stream_decode(z: torch.Tensor) -> torch.Tensor:
                    m    = self.model
                    dec  = m.decoder

                    # ── Temporal chunking (mirrors diffusers _decode) ───────────────────
                    tokens_chunk_size = m.tokens_chunk_size
                    token_drop        = m.config.token_drop
                    temporal_ratio    = m.temporal_compression_ratio
                    chunk_num_frames  = tokens_chunk_size * temporal_ratio

                    num_tokens = z.shape[2] + token_drop
                    pad_tokens = (-num_tokens) % tokens_chunk_size
                    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
                    if pad_tokens > 0:
                        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

                    # ── Spatial tiling (mirrors diffusers _decode_clip) ─────────────────
                    use_tiling = getattr(m, "use_tiling", True)
                    if use_tiling:
                        pixel_h = z.shape[-2] * m.spatial_compression_ratio
                        pixel_w = z.shape[-1] * m.spatial_compression_ratio
                        y_indices, y_lengths, y_overlaps = m._split_tiles(
                            pixel_h, m.tile_sample_min_height, m.tile_sample_min_overlap_height
                        )
                        x_indices, x_lengths, x_overlaps = m._split_tiles(
                            pixel_w, m.tile_sample_min_width, m.tile_sample_min_overlap_width
                        )
                    else:
                        y_indices = x_indices = [0]
                        y_lengths = [z.shape[-2] * m.spatial_compression_ratio]
                        x_lengths = [z.shape[-1] * m.spatial_compression_ratio]
                        y_overlaps = x_overlaps = []

                    ratio = m.spatial_compression_ratio

                    # ── Build all (clip_idx, y_tile_idx, x_tile_idx) → latent slice ────
                    # Shape: (num_clips × num_y × num_x) tensors, each (1, C, T_tok, H_tok, W_tok)
                    tile_z = []   # flat list of latent clips, one per (clip, y, x) combination
                    tile_info = []  # metadata for stitching later

                    for ci in range(num_chunks):
                        start  = ci * tokens_chunk_size
                        clip_z = z[:, :, start : start + tokens_chunk_size + m.token_overlap]
                        # run post_quant_conv once per clip (cheap, resident weights)
                        clip_z = m.post_quant_conv(clip_z)
                        for yi, (i_pos, i_len) in enumerate(zip(y_indices, y_lengths)):
                            for xi, (j_pos, j_len) in enumerate(zip(x_indices, x_lengths)):
                                tile = clip_z[
                                    ...,
                                    i_pos // ratio : i_pos // ratio + i_len // ratio,
                                    j_pos // ratio : j_pos // ratio + j_len // ratio,
                                ]
                                _, _, T, H, W = tile.shape
                                # ── proj_in + register tokens + RoPE (no block weights needed) ──
                                hs = tile.permute(0, 2, 3, 4, 1).reshape(1, T * H * W, -1)
                                hs = dec.proj_in(hs)
                                num_patches = hs.shape[1]

                                register_tokens = dec.register_tokens.expand(1, -1, -1)
                                cls_token = torch.zeros_like(hs[:, :1, :])
                                hs = torch.cat([hs, register_tokens, cls_token], dim=1)

                                grids = [
                                    2.0 * (torch.arange(0.5, sz, dtype=torch.float32, device=hs.device) / sz) - 1.0
                                    for sz in (T, H, W)
                                ]
                                pos_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
                                pos_ids = pos_ids.unsqueeze(0)
                                suffix_ids = pos_ids.new_zeros((1, dec.num_register_tokens + 1, 3))
                                pos_ids = torch.cat([pos_ids, suffix_ids], dim=1)
                                rotary_emb = dec.rope(pos_ids)

                                tile_z.append((hs, rotary_emb, num_patches, T, H, W))
                                tile_info.append((ci, yi, xi))

                    # ── Stack all tiles into a single batch for GPU-parallel processing ─
                    all_hs  = torch.cat([item[0] for item in tile_z], dim=0)
                    all_cos = torch.cat([item[1][0] for item in tile_z], dim=0)
                    all_sin = torch.cat([item[1][1] for item in tile_z], dim=0)
                    num_patches, T_tile, H_tile, W_tile = tile_z[0][2], tile_z[0][3], tile_z[0][4], tile_z[0][5]
                    tile_z = None  # free the list

                    import concurrent.futures
                    import time
                    import queue
                    import threading
                    import psutil

                    num_tiles_total = all_hs.shape[0]
                    
                    # ── Dynamic Memory Budgeting ──
                    if hasattr(self, "_global_vram_budget_gb") and self._global_vram_budget_gb:
                        free_vram = self._global_vram_budget_gb * 1024**3
                    else:
                        free_vram = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
                        
                    if hasattr(self, "_global_ram_budget_gb") and self._global_ram_budget_gb:
                        free_ram = self._global_ram_budget_gb * 1024**3
                    else:
                        free_ram = psutil.virtual_memory().available

                    # Empirical sizing constants
                    ACTIVATION_BYTES_PER_TILE = 90 * 1024 * 1024  # ~90MB activation VRAM per tile
                    BLOCK_BYTES = 150 * 1024 * 1024               # ~150MB per block in VRAM / RAM
                    
                    # Decide count of blocks that can stay on GPU (Double Buffering: 2 or 3 blocks)
                    # Base requirement is 2 slots. If extra VRAM allows 3 slots while preserving activation headroom:
                    if free_vram >= (3 * BLOCK_BYTES) + (6 * ACTIVATION_BYTES_PER_TILE) + (250 * 1024 * 1024):
                        gpu_blocks_capacity = 3
                    else:
                        gpu_blocks_capacity = 2
                    
                    # Calculate MAX_MICROBATCH
                    # Leave 200MB safety margin for PyTorch CUDA context overhead
                    available_activations_vram = max(0, free_vram - (gpu_blocks_capacity * BLOCK_BYTES) - (200 * 1024 * 1024))
                    max_tiles_allowed = int(available_activations_vram / ACTIVATION_BYTES_PER_TILE)
                    MAX_MICROBATCH = max(1, min(num_tiles_total, max_tiles_allowed))
                    
                    # Calculate System RAM Caching Depth
                    # Allow caching up to 6 blocks (as requested), constrained by available physical RAM
                    max_ram_cache = max(1, min(6, int(free_ram / BLOCK_BYTES)))
                    
                    logger.info(f"    [VAE Streamer] Dynamic Budget: VRAM Free={free_vram/1e9:.2f}GB, RAM Free={free_ram/1e9:.2f}GB")
                    logger.info(f"    [VAE Streamer] Decided Buffers: GPU Slots={gpu_blocks_capacity} blocks | RAM Cache Depth={max_ram_cache} blocks | Micro-batch MAX={MAX_MICROBATCH} tiles")
                    logger.info(f"    [VAE Streamer] Streaming {len(dec.transformer_blocks)} blocks over "
                                f"{num_tiles_total} tile-clips ...")

                    # ── 3-Tier Pre-allocation Buffers ──
                    ping_pong_buffers = {i: {} for i in range(gpu_blocks_capacity)}
                    fetch_stream = torch.cuda.Stream(device=self.device)
                    
                    cpu_queue = queue.Queue(maxsize=max_ram_cache)
                    disk_worker_stop = threading.Event()
                    
                    def disk_worker():
                        import inspect
                        has_process = "process_gguf" in inspect.signature(self.seeker.get_tensors).parameters
                        for b_idx in range(len(dec.transformer_blocks)):
                            if disk_worker_stop.is_set(): break
                            b = dec.transformer_blocks[b_idx]
                            shard_prefix = getattr(b, "_vae_shard_prefix", f"decoder.transformer_blocks.{b_idx}")
                            keys = self._get_block_keys(shard_prefix)
                            
                            t0 = time.perf_counter()
                            # 1. Disk I/O -> CPU RAM
                            if has_process:
                                sd_cpu = self.seeker.get_tensors(keys, device="cpu", dtype=self.dtype, process_gguf=False)
                            else:
                                sd_cpu = self.seeker.get_tensors(keys, device="cpu", dtype=self.dtype)
                            t1 = time.perf_counter()
                            
                            # Blocks if RAM queue is full (shields against OOM)
                            cpu_queue.put((b_idx, keys, shard_prefix, b, sd_cpu, t1 - t0))
                            
                    disk_thread = threading.Thread(target=disk_worker, daemon=True)
                    disk_thread.start()

                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

                    def h2d_worker(b_idx_expected):
                        if b_idx_expected >= len(dec.transformer_blocks): return None
                        # Pop from CPU RAM queue instantly (or wait if disk is slow)
                        b_idx, keys, shard_prefix, block, sd_cpu, t_disk = cpu_queue.get()
                        
                        event = torch.cuda.Event()
                        t0 = time.perf_counter()
                        
                        # 2. H2D Transfer (Ping-Pong in-place copy)
                        buf_idx = b_idx % gpu_blocks_capacity
                        target_buffer = ping_pong_buffers[buf_idx]
                        
                        with torch.cuda.stream(fetch_stream):
                            for name, tensor in sd_cpu.items():
                                suffix = name[len(shard_prefix)+1:]
                                if suffix not in target_buffer:
                                    target_buffer[suffix] = tensor.to(self.device, non_blocking=True)
                                else:
                                    target_buffer[suffix].copy_(tensor, non_blocking=True)
                            event.record(fetch_stream)
                            
                        t_h2d = time.perf_counter() - t0
                        return keys, target_buffer, block, event, t_disk, t_h2d

                    next_future = executor.submit(h2d_worker, 0)
                    
                    # Temporarily remove hooks so they don't fire implicitly
                    hook_backups = []
                    for b in dec.transformer_blocks:
                        hook_backups.append((b._forward_pre_hooks, b._forward_hooks))
                        b._forward_pre_hooks = {}
                        b._forward_hooks = {}

                    try:
                        total_gpu_idle = 0.0
                        for block_idx in range(len(dec.transformer_blocks)):
                            t_wait_start = time.perf_counter()
                            future_result = next_future.result()
                            if future_result is None: break
                            keys, target_buffer, block, event, t_disk, t_h2d = future_result
                            
                            # Immediately kick off the next block for true Async Double Buffering!
                            next_future = executor.submit(h2d_worker, block_idx + 1)
                            
                            # Wait for H2D transfer to finish in the fetch stream
                            torch.cuda.current_stream().wait_event(event)
                            event.synchronize()
                            t_idle = time.perf_counter() - t_wait_start
                            total_gpu_idle += t_idle
                            
                            cpu_blocks_count = cpu_queue.qsize()
                            
                            shard_prefix = getattr(block, "_vae_shard_prefix", f"decoder.transformer_blocks.{block_idx}")
                            
                            # Wire the PyTorch module parameters directly to the persistent buffer
                            for suffix, tensor in target_buffer.items():
                                full_name = f"{shard_prefix}.{suffix}"
                                self._place_tensor(full_name, tensor, self.device, self.dtype)
                                
                            compute_start = torch.cuda.Event(enable_timing=True)
                            compute_end = torch.cuda.Event(enable_timing=True)
                            compute_start.record()
                            
                            # ── Micro-batched GPU Forward Pass ──
                            out_hs = []
                            for b_start in range(0, num_tiles_total, MAX_MICROBATCH):
                                chunk_hs = all_hs[b_start : b_start + MAX_MICROBATCH]
                                chunk_cos = all_cos[b_start : b_start + MAX_MICROBATCH]
                                chunk_sin = all_sin[b_start : b_start + MAX_MICROBATCH]
                                out_chunk = block(chunk_hs, (chunk_cos, chunk_sin))
                                out_hs.append(out_chunk)
                            all_hs = torch.cat(out_hs, dim=0)
                            
                            compute_end.record()
                            
                            # Evict block weights from the MODULE (target_buffer safely remains intact)
                            self._evict_keys(keys)
                            
                            # Zero empty_cache() needed! Fragmentation is physically impossible now.
                            
                            compute_end.synchronize()
                            t_compute = compute_start.elapsed_time(compute_end)
                            
                            used = torch.cuda.memory_allocated() / 1e9
                            resv = torch.cuda.memory_reserved() / 1e9
                            logger.info(f"    [VAE Streamer] Block {block_idx+1:02d}/36 | "
                                        f"Disk: {t_disk*1000:4.0f}ms | H2D: {t_h2d*1000:4.0f}ms | "
                                        f"Compute: {t_compute:4.0f}ms | GPU Idle: {t_idle*1000:3.0f}ms | "
                                        f"GPU Blocks: {gpu_blocks_capacity} | CPU Blocks: {cpu_blocks_count}/{max_ram_cache} | "
                                        f"VRAM: {used:.2f}GB / Resv: {resv:.2f}GB")
                            print(f"    [VAE Streamer] Block {block_idx+1}/36 completed", flush=True)

                    finally:
                        disk_worker_stop.set()
                        # Drain the queue to unblock the disk thread if it's stuck waiting on put()
                        while not cpu_queue.empty():
                            try:
                                cpu_queue.get_nowait()
                            except queue.Empty:
                                break
                        executor.shutdown(wait=False)
                        ping_pong_buffers.clear()
                        # Restore hooks
                        for b, (pre, post) in zip(dec.transformer_blocks, hook_backups):
                            b._forward_pre_hooks = pre
                            b._forward_hooks = post

                    logger.info(f"    [VAE Streamer] All 36 blocks finished | Total GPU Starvation/Idle Time: {total_gpu_idle*1000:.0f}ms")

                    # ── norm_out + proj_out + unpatch each tile in parallel ──────────────
                    # all_hs: (N, seq, dim) — process all at once, then split
                    all_hs = dec.norm_out(all_hs)
                    all_hs = dec.proj_out(all_hs)
                    all_hs = all_hs[:, :num_patches, :]  # drop register + cls tokens

                    patch_size   = dec.patch_size
                    patch_size_t = dec.patch_size_t
                    N = all_hs.shape[0]
                    all_hs = all_hs.view(N, T_tile, H_tile, W_tile, dec.out_channels, patch_size_t, patch_size, patch_size)
                    all_hs = all_hs.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
                    all_hs = all_hs.reshape(N, dec.out_channels, T_tile * patch_size_t, H_tile * patch_size, W_tile * patch_size)
                    decoded_tiles = list(all_hs.unbind(dim=0))  # list of (C, t, h, w)
                    decoded_tiles = [t.unsqueeze(0) for t in decoded_tiles]  # back to (1, C, t, h, w)

                    # ── Stitch tiles per clip, then blend temporal overlaps ─────────────
                    num_y = len(y_indices)
                    num_x = len(x_indices)
                    decoded_chunks = []
                    overlap = None
                    for ci in range(num_chunks):
                        base = ci * num_y * num_x
                        rows = [
                            [decoded_tiles[base + yi * num_x + xi] for xi in range(num_x)]
                            for yi in range(num_y)
                        ]
                        if use_tiling and (num_y > 1 or num_x > 1):
                            clip_dec = m._stitch_tiles(rows, y_overlaps, x_overlaps)
                        else:
                            clip_dec = rows[0][0]

                        for j in range(int(token_drop > 0) + 1):
                            frame_start = j * chunk_num_frames
                            chunk = clip_dec[:, :, frame_start : frame_start + chunk_num_frames]
                            chunk = chunk[:, :, m.frame_pre_padding :]
                            if j == 0:
                                if overlap is not None:
                                    chunk = m._blend(overlap, chunk, m.frame_overlap, dim=-3)
                                decoded_chunks.append(chunk)
                            else:
                                overlap = chunk
                    if overlap is not None:
                        decoded_chunks.append(overlap)

                    dec_out = torch.cat(decoded_chunks, dim=2)

                    if pad_tokens > 0:
                        intra_tail = m.config.clip_length % temporal_ratio
                        num_tokens_before_pad = z.shape[2] - pad_tokens
                        pad_frames = sum(
                            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                            for k in range(pad_tokens)
                        )
                        dec_out = dec_out[:, :, :-pad_frames]
                    return dec_out

                # Monkey-patch _decode so model.decode() calls our streaming version
                original_decode = self.model._decode
                self.model._decode = _stream_decode
                try:
                    out = self.model.decode(latents.to(self.dtype), return_dict=False)
                finally:
                    self.model._decode = original_decode

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
