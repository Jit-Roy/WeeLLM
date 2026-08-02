"""
streaming_pipeline.py -- WeeFlux2KleinPipeline

Memory-frugal inference pipeline for FLUX.2 [klein] 4B.
Fits under 4 GB VRAM and 8 GB RAM by streaming every heavy component
one layer at a time.

Architecture
------------
  text_encoder (Qwen3, 36 layers, ~8 GB)   -> streamed layer-by-layer
  transformer  (Flux2, 25 blocks, ~7.2 GB)  -> streamed block-by-block
  vae          (~160 MB)                     -> resident (always on GPU)
  scheduler + tokenizer                      -> tiny, always in RAM

Usage
-----
from weellm import WeeFlux2KleinPipeline
import torch

pipe = WeeFlux2KleinPipeline.from_pretrained(
    "D:/Personal Projects/WeeLLM/flux2-klein-4b",
    device="cuda",
    dtype=torch.bfloat16,
)
image = pipe.generate(
    prompt="A cat holding a sign that says hello world",
    height=512,
    width=512,
    num_inference_steps=4,
    seed=42,
)
image.save("output.png")
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .transformer_streamer import FluxStreamer
from .text_encoder_streamer import StreamingQwen3TextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


class WeeFlux2KleinPipeline(BasePipeline):
    """
    Memory-frugal FLUX.2 Klein 4B pipeline.

    Streams every heavy component (text encoder + transformer) one layer
    at a time so the peak VRAM never exceeds ~1.5 GB during generation.

    Parameters
    ----------
    model_dir : Path
        Root directory of the Flux2 Klein model checkout.
    device : str
        Target device.
    dtype : torch.dtype
        Compute dtype (bfloat16 recommended).
    prefetch : bool
        Enable background prefetching of the next transformer block while
        the current one runs (overlaps disk I/O with GPU compute).
    max_text_length : int
        Maximum tokenised prompt length (default 512, matching the pipeline).
    extract_layers : tuple[int]
        Qwen3 layers whose hidden states are combined for the prompt embedding
        (default: (9, 18, 27) -- as used by the original Flux2KleinPipeline).
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        max_text_length: int = 512,
        extract_layers: tuple = (9, 18, 27),
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch
        self.max_text_length = max_text_length
        self.extract_layers = extract_layers

        self._transformer_streamer: Optional[FluxStreamer] = None
        self._text_encoder: Optional[StreamingQwen3TextEncoder] = None
        self._vae = None
        self._scheduler = None
        self._vae_scale_factor = 8  # Flux2 VAE: 8x spatial compression

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        max_text_length: int = 512,
        extract_layers: tuple = (9, 18, 27),
        force_resplit: bool = False,
    ) -> "WeeFlux2KleinPipeline":
        """Initialise the streaming pipeline from a local model directory."""
        model_dir = Path(model_dir)
        pipe = cls(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
            max_text_length=max_text_length,
            extract_layers=extract_layers,
        )
        pipe._load_components(force_resplit=force_resplit)
        return pipe

    # ------------------------------------------------------------------
    # Component loading
    # ------------------------------------------------------------------

    def _load_components(self, force_resplit: bool = False):
        print("\n" + "=" * 60)
        print("  WeeFlux2KleinPipeline -- initialising")
        print("=" * 60)

        # ---- Scheduler (tiny) ----------------------------------------
        print("\n[1/4] Loading scheduler ...")
        from diffusers import FlowMatchEulerDiscreteScheduler
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(self.model_dir / "scheduler")
        )
        print("      OK")

        # ---- VAE (~160 MB -- kept resident on GPU) --------------------
        print("\n[2/4] Loading VAE (~160 MB, resident on GPU) ...")
        from diffusers import AutoencoderKLFlux2
        self._vae = AutoencoderKLFlux2.from_pretrained(
            str(self.model_dir / "vae"), torch_dtype=self.dtype
        ).to(self.device)
        self._vae.eval()
        clean_memory(self.device)
        report_memory("After VAE load")

        # ---- Text encoder (streaming) --------------------------------
        print("\n[3/4] Preparing streaming Qwen3 text encoder ...")
        self._text_encoder = StreamingQwen3TextEncoder(
            text_encoder_dir=self.model_dir / "text_encoder",
            tokenizer_dir=self.model_dir / "tokenizer",
            device=self.device,
            dtype=self.dtype,
            extract_layers=self.extract_layers,
            max_length=self.max_text_length,
        )
        # Trigger initialization (splits shards on first run)
        self._text_encoder._ensure_initialized()
        report_memory("After text encoder init")

        # ---- Transformer (streaming) ---------------------------------
        print("\n[4/4] Preparing streaming Flux2 transformer ...")
        self._transformer_streamer = FluxStreamer.from_pretrained(
            transformer_dir=self.model_dir / "transformer",
            device=self.device,
            dtype=self.dtype,
            prefetch=self.prefetch,
            force_resplit=force_resplit,
        )
        report_memory("After transformer init")

        print("\n" + "=" * 60)
        print("  Pipeline ready.")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Latent packing / unpacking helpers (from Flux2KleinPipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> latent_ids (B, H*W, 4) with [t, h, w, l] coords."""
        B, _, H, W = latents.shape
        t = torch.arange(1)
        h = torch.arange(H)
        w = torch.arange(W)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, h, w, l)  # (H*W, 4)
        return ids.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> text_ids (B, L, 4) with [t=0, h=0, w=0, l=i] coords."""
        B, L, _ = prompt_embeds.shape
        out_ids = []
        for _ in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)  # (L, 4)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def _unpack_latents(
        x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Scatter sequence tokens back into (B, C, H, W) using position IDs."""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].long()
            w_ids = pos[:, 2].long()
            flat_ids = h_ids * width + w_ids
            out = torch.zeros((height * width, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            out = out.view(height, width, ch).permute(2, 0, 1)
            x_list.append(out)
        return torch.stack(x_list, dim=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Generate an image from a text prompt.

        Parameters
        ----------
        prompt : str
            Text description of the desired image.
        height, width : int
            Output dimensions (multiples of 16 recommended).
        num_inference_steps : int
            Number of denoising steps (4 for this distilled model).
        guidance_scale : float
            CFG scale.  1.0 disables classifier-free guidance (saves 2x
            transformer passes per step).
        seed : int, optional
            RNG seed for reproducibility.
        output_type : str
            "pil" -> PIL.Image.Image, "tensor" -> raw latents (B, C, H, W).

        Returns
        -------
        image : PIL.Image.Image or torch.Tensor
        """
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

        # ---- 1. Encode prompt ----------------------------------------
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds = self._text_encoder.encode(prompt)
        # prompt_embeds shape: (1, max_text_length, 7680)
        clean_memory(self.device)
        report_memory("After text encode")

        text_ids = self._prepare_text_ids(prompt_embeds).to(self.device)

        # ---- 2. Prepare latents --------------------------------------
        # Flux2 latent space: VAE 8x compression + 2x packing factor
        # Raw latent shape: (B, num_channels, H//8, W//8)
        # After packing:    (B, (H//8) * (W//8), num_channels * 4)
        # num_latent_channels = transformer.in_channels // 4 = 128 // 4 = 32

        # Use Flux2 convention:
        # effective_h = 2 * (height // (vae_scale_factor * 2))
        # effective_w = 2 * (width  // (vae_scale_factor * 2))
        eff_h = 2 * (int(height) // (self._vae_scale_factor * 2))
        eff_w = 2 * (int(width)  // (self._vae_scale_factor * 2))
        num_latent_channels = 32   # in_channels(128) // 4

        # shape = (B, num_latent_channels * 4, eff_h // 2, eff_w // 2)
        # = (1, 128, eff_h//2, eff_w//2)
        raw_latent_shape = (1, num_latent_channels * 4, eff_h // 2, eff_w // 2)
        latents_raw = torch.randn(
            raw_latent_shape, dtype=self.dtype, device=self.device, generator=generator
        )
        latent_ids = self._prepare_latent_ids(latents_raw).to(self.device)
        latents = self._pack_latents(latents_raw)  # (1, H*W, C)
        del latents_raw

        # ---- 3. Denoising loop ---------------------------------------
        print("  [2/3] Denoising ...")

        # Flux2 uses empirical mu for timestep shifting
        image_seq_len = latents.shape[1]
        try:
            from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
            mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                self._scheduler, num_inference_steps, self.device, sigmas=sigmas, mu=mu
            )
        except Exception:
            # Fallback: simple uniform timesteps
            self._scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self._scheduler.timesteps

        for step_idx, t in enumerate(timesteps):
            print(f"        Step {step_idx + 1}/{num_inference_steps} (t={t.item():.4f}) ...")
            report_memory(f"  step {step_idx+1}")

            latent_model_input = latents.to(self.dtype)
            timestep = t.expand(1).to(self.dtype)

            # Run streaming transformer
            # NOTE: timestep is divided by 1000 (as the Flux2 pipeline does)
            noise_pred = self._transformer_streamer.model(
                hidden_states=latent_model_input,
                timestep=timestep / 1000.0,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                guidance=None,
                return_dict=False,
            )[0]

            # Scheduler step
            latents = self._scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

            clean_memory(self.device)

        if output_type == "tensor":
            return latents

        # ---- 4. VAE decode -------------------------------------------
        # Follows the exact Flux2KleinPipeline postprocessing sequence:
        #   (a) Unpack sequence tokens -> (B, C, H, W)  [128 channels]
        #   (b) BN denorm: latents * bn_std + bn_mean
        #   (c) _unpatchify_latents: (B, 128, H//2, W//2) -> (B, 32, H, W)
        #   (d) vae.decode(latents)
        print("  [3/3] Decoding ...")
        latent_h = eff_h // 2
        latent_w = eff_w // 2

        # (a) Unpack packed sequence -> spatial latent map
        latents_spatial = self._unpack_latents(latents, latent_ids, latent_h, latent_w)
        # latents_spatial: (1, 128, latent_h, latent_w)

        # (b) Batch-norm denormalization (Flux2Klein-specific normalization)
        bn_mean = self._vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latents_spatial.device, dtype=latents_spatial.dtype
        )
        bn_std = torch.sqrt(
            self._vae.bn.running_var.view(1, -1, 1, 1)
            + self._vae.config.get("batch_norm_eps", 1e-4)
        ).to(device=latents_spatial.device, dtype=latents_spatial.dtype)
        latents_spatial = latents_spatial * bn_std + bn_mean

        # (c) Unpatchify: (B, C*4, H//2, W//2) -> (B, C, H, W)
        latents_spatial = self._unpatchify_latents(latents_spatial)
        # latents_spatial: (1, 32, latent_h*2, latent_w*2) = (1, 32, eff_h, eff_w)

        # (d) VAE decode
        latents_spatial = latents_spatial.to(dtype=self.dtype)
        image_tensor = self._vae.decode(latents_spatial, return_dict=False)[0]
        clean_memory(self.device)
        report_memory("After VAE decode")

        # Postprocess: [-1, 1] -> [0, 1] -> uint8
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor[0].cpu().float().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().astype("uint8")
        image = Image.fromarray(image_np)

        print("  Done!\n")
        return image

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Reverse the 2x2 spatial packing applied before the transformer.
        (B, C*4, H, W) -> (B, C, H*2, W*2)
        Matches Flux2KleinPipeline._unpatchify_latents exactly.
        """
        B, C4, H, W = latents.shape
        C = C4 // 4
        latents = latents.reshape(B, C, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C, H * 2, W * 2)
        return latents

