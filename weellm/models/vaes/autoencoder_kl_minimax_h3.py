import json
import logging
import threading
import queue
import time
import concurrent.futures
from pathlib import Path
from typing import List, Union

import psutil
import torch
import torch.nn as nn
from accelerate import init_empty_weights

from weellm.io.seeker import get_seeker
from weellm.io.utils import default_dtype, clean_memory, report_memory
from .base_vae_streamer import BaseVAEStreamer

logger = logging.getLogger("weellm")


class AutoencoderKLMiniMaxH3Streamer(BaseVAEStreamer):
    """
    Memory-efficient streaming VAE wrapper for the MiniMax-H3 Video VAE.

    Architecture overview
    ---------------------
    The MiniMaxH3 decoder is a 36-block ViT that operates on a packed sequence of
    (temporal_clip x spatial_tile) patch tokens.  Loading all 36 blocks at once
    would require ~5 GB of VRAM -- far beyond a 4 GB budget.

    Strategy: 3-tier async pipeline
      Tier 1 - Disk thread: continuously reads the next block shard from disk
               into a CPU-RAM queue (prefetch depth = max_ram_cache).
      Tier 2 - H2D worker (ThreadPoolExecutor): pulls a block from the CPU queue,
               pins it to a ping-pong GPU buffer, submits an async H2D copy on a
               dedicated CUDA stream.
      Tier 3 - GPU compute stream: waits for the H2D event, wires the buffer into
               nn.Module params via _place_tensor, runs the micro-batched forward
               over all tiles in-place, evicts the weights, then kicks off the
               next H2D transfer while compute is in flight.

    Only 2 GPU block buffers (ping-pong) are kept alive at any time, so peak
    VRAM from block weights is always 2 x ~150 MB = ~300 MB.

    decode() also handles:
      - Temporal chunking  (mirrors diffusers _decode)
      - Spatial tiling     (mirrors diffusers _decode_clip / _split_tiles)
      - Temporal overlap blending
      - Result caching to .weellm_cache/vae_decode_cache.pt
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
        self._config_extras = config_extras or {}
        self._patch_encode()

    @property
    def decoder_streaming_prefixes(self) -> tuple:
        return ("decoder.transformer_blocks.",)

    def _patch_encode(self) -> None:
        """Wrap model.encode() to lazy-load encoder weights on first call."""
        original_encode = self.model.encode

        def _lazy_encode(self_obj, *args, **kwargs):
            self._load_encoder()
            kwargs.pop("return_dict", None)
            if args and isinstance(args[0], torch.Tensor):
                args = (args[0].to(self.dtype),) + args[1:]
            result = original_encode(*args, return_dict=True, **kwargs)
            if hasattr(result, "latent_dist"):
                posterior = result.latent_dist
            else:
                from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
                posterior = DiagonalGaussianDistribution(result)
            self._evict_encoder()
            return (posterior,)

        self.model.encode = _lazy_encode.__get__(self.model, self.model.__class__)

    def decode(self, latents, return_dict=True, **kwargs):
        """Decode latents through the 3-tier streaming VAE decoder.

        Loads one of the 36 transformer blocks at a time using an async
        disk->CPU->GPU pipeline, keeping peak VRAM to ~2x block_size at all times.
        Results are cached to disk to avoid re-decoding on crash/resume.
        """
        import os
        cache_dir = os.path.join(os.getcwd(), ".weellm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "vae_decode_cache.pt")

        if os.path.exists(cache_file):
            logger.info("    [VAE Streamer] Loading cached decoded video from %s ...", cache_file)
            result = torch.load(cache_file, map_location=latents.device)
            logger.info("    [VAE Streamer] Cached decode loaded: %s", tuple(result.shape))
        else:
            with torch.no_grad():
                report_memory("Before VAE decode")
                result = self._stream_decode(latents.to(self.dtype))
                report_memory("After VAE decode")
                logger.info("    [VAE Streamer] decode done: %s", tuple(result.shape))

            logger.info("    [VAE Streamer] Saving video decode cache to %s ...", cache_file)
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

    def _stream_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Full 3-tier streaming decode:
          1. Pre-processes all temporal clips x spatial tiles into a batched
             hidden-state tensor (all_hs).
          2. Streams each of the 36 ViT decoder blocks through the async
             disk->CPU->GPU pipeline, running micro-batched forwards in-place.
          3. Applies norm_out, proj_out, unpatch, stitch, temporal blending.
        """
        m   = self.model
        dec = m.decoder

        # Temporal chunking (mirrors diffusers _decode)
        tokens_chunk_size = m.tokens_chunk_size
        token_drop        = m.config.token_drop
        temporal_ratio    = m.temporal_compression_ratio
        chunk_num_frames  = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        # Spatial tiling (mirrors diffusers _decode_clip)
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

        # Build all (clip, y_tile, x_tile) -> hidden states
        tile_z    = []
        tile_info = []

        for ci in range(num_chunks):
            start  = ci * tokens_chunk_size
            clip_z = z[:, :, start : start + tokens_chunk_size + m.token_overlap]
            # post_quant_conv is resident on GPU - cheap, no disk I/O
            clip_z = m.post_quant_conv(clip_z)
            for yi, (i_pos, i_len) in enumerate(zip(y_indices, y_lengths)):
                for xi, (j_pos, j_len) in enumerate(zip(x_indices, x_lengths)):
                    tile = clip_z[
                        ...,
                        i_pos // ratio : i_pos // ratio + i_len // ratio,
                        j_pos // ratio : j_pos // ratio + j_len // ratio,
                    ]
                    _, _, T, H, W = tile.shape
                    # proj_in + register tokens + RoPE -- all resident, no disk I/O
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

        # Stack all tiles into one batch for blocked GPU processing
        all_hs  = torch.cat([item[0] for item in tile_z], dim=0)
        all_cos = torch.cat([item[1][0] for item in tile_z], dim=0)
        all_sin = torch.cat([item[1][1] for item in tile_z], dim=0)
        num_patches, T_tile, H_tile, W_tile = tile_z[0][2], tile_z[0][3], tile_z[0][4], tile_z[0][5]
        tile_z = None  # free list

        num_tiles_total = all_hs.shape[0]

        # Dynamic Memory Budgeting
        # free_vram is queried AFTER all_hs is already in VRAM, so it is the
        # headroom beyond all_hs + other resident tensors.
        free_vram = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        if hasattr(self, "_global_ram_budget_gb") and self._global_ram_budget_gb:
            free_ram = self._global_ram_budget_gb * 1024 ** 3
        else:
            free_ram = psutil.virtual_memory().available

        # Measure actual per-tile size from the live tensor (no guessing)
        TILE_BYTES  = all_hs.nbytes // max(num_tiles_total, 1)
        BLOCK_BYTES = 150 * 1024 * 1024  # ~150 MB per block weights in VRAM

        # 2 ping-pong GPU slots saves 150 MB vs 3 slots.
        # Peak per-tile VRAM during forward:
        #   all_hs (held) + out_chunk (1x) + SwiGLU intermediates (~4x) = ~5x TILE_BYTES
        gpu_blocks_capacity      = 2
        OVERHEAD_BYTES           = 300 * 1024 * 1024  # 300 MB safety margin
        available_for_microbatch = max(0, free_vram - gpu_blocks_capacity * BLOCK_BYTES - OVERHEAD_BYTES)
        max_tiles_allowed = max(1, int(available_for_microbatch // (5 * TILE_BYTES))) if TILE_BYTES > 0 else 1
        MAX_MICROBATCH    = max(1, min(num_tiles_total, max_tiles_allowed))

        # RAM cache depth: number of blocks to prefetch into CPU RAM
        max_ram_cache = max(1, min(6, int(free_ram / BLOCK_BYTES)))

        logger.info(
            "    [VAE Streamer] Dynamic Budget: VRAM Free=%.2fGB, RAM Free=%.2fGB | tile_bytes=%.1fMB",
            free_vram / 1e9, free_ram / 1e9, TILE_BYTES / 1e6,
        )
        logger.info(
            "    [VAE Streamer] Decided Buffers: GPU Slots=%d blocks | RAM Cache Depth=%d blocks | "
            "Micro-batch MAX=%d/%d tiles",
            gpu_blocks_capacity, max_ram_cache, MAX_MICROBATCH, num_tiles_total,
        )
        logger.info(
            "    [VAE Streamer] Streaming %d blocks over %d tile-clips ...",
            len(dec.transformer_blocks), num_tiles_total,
        )

        # 3-Tier pre-allocation
        ping_pong_buffers = {i: {} for i in range(gpu_blocks_capacity)}
        fetch_stream      = torch.cuda.Stream(device=self.device)
        cpu_q             = queue.Queue(maxsize=max_ram_cache)
        disk_stop         = threading.Event()

        def disk_worker():
            """Tier 1: Read block shards from disk into CPU RAM."""
            for blk_idx, block in enumerate(dec.transformer_blocks):
                if disk_stop.is_set():
                    break
                shard_prefix = getattr(block, "_vae_shard_prefix",
                                       f"decoder.transformer_blocks.{blk_idx}")
                keys = self._get_block_keys(shard_prefix)
                t0 = time.perf_counter()
                sd_cpu = self.seeker.get_tensors(keys, device="cpu", dtype=None)
                t_disk = time.perf_counter() - t0
                cpu_q.put((blk_idx, keys, block, shard_prefix, sd_cpu, t_disk))
            cpu_q.put(None)  # sentinel

        disk_thread = threading.Thread(target=disk_worker, daemon=True)
        disk_thread.start()

        def h2d_worker(b_idx):
            """Tier 2: Async H2D copy from CPU RAM to ping-pong GPU buffer."""
            item = cpu_q.get()
            if item is None:
                return None
            blk_idx, keys, block, shard_prefix, sd_cpu, t_disk = item
            target_buffer = ping_pong_buffers[b_idx % gpu_blocks_capacity]
            t0 = time.perf_counter()
            event = torch.cuda.Event()
            with torch.cuda.stream(fetch_stream):
                for name, tensor in sd_cpu.items():
                    suffix = name[len(shard_prefix) + 1:]
                    if suffix not in target_buffer:
                        target_buffer[suffix] = tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)
                    else:
                        target_buffer[suffix].copy_(tensor, non_blocking=True)
                event.record(fetch_stream)
            t_h2d = time.perf_counter() - t0
            return keys, target_buffer, block, shard_prefix, event, t_disk, t_h2d

        # Safety: suppress any lingering forward hooks on the blocks so they cannot
        # fire during the manual forward calls below.
        hook_backups = []
        for b in dec.transformer_blocks:
            hook_backups.append((b._forward_pre_hooks.copy(), b._forward_hooks.copy()))
            b._forward_pre_hooks.clear()
            b._forward_hooks.clear()

        try:
            total_gpu_idle = 0.0
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                next_future = executor.submit(h2d_worker, 0)

                for block_idx in range(len(dec.transformer_blocks)):
                    t_wait_start  = time.perf_counter()
                    future_result = next_future.result()
                    if future_result is None:
                        break
                    keys, target_buffer, block, shard_prefix, event, t_disk, t_h2d = future_result

                    # Kick off next block H2D immediately (true async overlap)
                    next_future = executor.submit(h2d_worker, block_idx + 1)

                    # Wait for this block H2D to complete
                    torch.cuda.current_stream().wait_event(event)
                    event.synchronize()
                    t_idle = time.perf_counter() - t_wait_start
                    total_gpu_idle += t_idle

                    cpu_blocks_count = cpu_q.qsize()

                    # Wire ping-pong buffer tensors into the nn.Module parameters
                    for suffix, tensor in target_buffer.items():
                        self._place_tensor(f"{shard_prefix}.{suffix}", tensor, self.device, self.dtype)

                    compute_start = torch.cuda.Event(enable_timing=True)
                    compute_end   = torch.cuda.Event(enable_timing=True)
                    compute_start.record()

                    # Micro-batched GPU forward pass (in-place update of all_hs)
                    # Writing output directly back into all_hs avoids a torch.cat
                    # allocation that would double peak VRAM.
                    for b_start in range(0, num_tiles_total, MAX_MICROBATCH):
                        chunk_hs  = all_hs [b_start : b_start + MAX_MICROBATCH]
                        chunk_cos = all_cos[b_start : b_start + MAX_MICROBATCH]
                        chunk_sin = all_sin[b_start : b_start + MAX_MICROBATCH]
                        out_chunk = block(chunk_hs, (chunk_cos, chunk_sin))
                        # In-place copy: reuses all_hs storage; out_chunk freed immediately
                        all_hs[b_start : b_start + MAX_MICROBATCH].copy_(out_chunk)
                        del out_chunk, chunk_hs, chunk_cos, chunk_sin
                        if hasattr(torch, "clear_autocast_cache"):
                            torch.clear_autocast_cache()
                        torch.cuda.empty_cache()

                    compute_end.record()

                    # Evict block weights back to meta
                    self._evict_keys(keys)
                    # _place_tensor may have allocated dtype-cast copies in the CUDA pool.
                    # empty_cache() forces immediate reclamation (~130 MB per block).
                    torch.cuda.empty_cache()

                    compute_end.synchronize()
                    t_compute = compute_start.elapsed_time(compute_end)

                    used = torch.cuda.memory_allocated() / 1e9
                    resv = torch.cuda.memory_reserved() / 1e9
                    logger.info(
                        "    [VAE Streamer] Block %02d/%d | Disk: %4.0fms | H2D: %4.0fms | "
                        "Compute: %4.0fms | GPU Idle: %3.0fms | GPU Blocks: %d | "
                        "CPU Blocks: %d/%d | VRAM: %.2fGB / Resv: %.2fGB",
                        block_idx + 1, len(dec.transformer_blocks),
                        t_disk * 1000, t_h2d * 1000, t_compute,
                        t_idle * 1000, gpu_blocks_capacity,
                        cpu_blocks_count, max_ram_cache, used, resv,
                    )

        finally:
            disk_stop.set()
            while not cpu_q.empty():
                try:
                    cpu_q.get_nowait()
                except queue.Empty:
                    break
            disk_thread.join(timeout=5)
            ping_pong_buffers.clear()
            # Restore any hooks that were present before streaming
            for b, (pre, post) in zip(dec.transformer_blocks, hook_backups):
                b._forward_pre_hooks = pre
                b._forward_hooks     = post

        logger.info(
            "    [VAE Streamer] All %d blocks finished | Total GPU Starvation/Idle Time: %.0fms",
            len(dec.transformer_blocks), total_gpu_idle * 1000,
        )

        # norm_out + proj_out + unpatch (all resident weights, no disk I/O)
        all_hs = dec.norm_out(all_hs)
        all_hs = dec.proj_out(all_hs)
        all_hs = all_hs[:, :num_patches, :]  # drop register + cls tokens

        patch_size   = dec.patch_size
        patch_size_t = dec.patch_size_t
        N = all_hs.shape[0]
        all_hs = all_hs.view(N, T_tile, H_tile, W_tile, dec.out_channels, patch_size_t, patch_size, patch_size)
        all_hs = all_hs.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        all_hs = all_hs.reshape(N, dec.out_channels, T_tile * patch_size_t, H_tile * patch_size, W_tile * patch_size)
        decoded_tiles = [t.unsqueeze(0) for t in all_hs.unbind(dim=0)]  # list of (1, C, t, h, w)

        # Stitch tiles per clip, then blend temporal overlaps
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
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                else temporal_ratio
                for k in range(pad_tokens)
            )
            dec_out = dec_out[:, :, :-pad_frames]

        return dec_out

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

        vae_dir = Path(vae_dir)
        config_path = vae_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on MiniMax VAE weights ...")
        source_path = vae_dir / config["source_path"] if "source_path" in config else vae_dir
        seeker = get_seeker(str(source_path), cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating AutoencoderKLMiniMaxH3 on meta device ...")
        init_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
        with init_empty_weights():
            model = AutoencoderKLMiniMaxH3(**init_cfg)
        model.eval()

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

        # Non-persistent buffers (e.g. decoder.rope.inv_freq) land on CPU under
        # init_empty_weights(). Move them to the target device so RoPE does not
        # crash with a cuda:0 vs cpu device mismatch.
        _decoder = getattr(model, "decoder", None)
        if _decoder is not None:
            for buf_name, buf in list(_decoder.named_buffers()):
                if buf.device.type == "cpu":
                    buf.data = buf.data.to(device)
                    logger.info("    [VAE] Moved decoder buffer '%s' -> %s", buf_name, device)

        clean_memory(device)
        report_memory("After VAE resident load")
        logger.info("  MiniMaxVAEStreamer ready (%d resident keys, decoder blocks streamed).", len(resident_keys))
        return streamer
