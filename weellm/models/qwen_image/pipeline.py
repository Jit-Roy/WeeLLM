"""
pipeline.py -- WeeQwenImagePipeline

Memory-frugal inference pipeline for Qwen-Image.

Architecture:
  - Text encoder: Qwen2_5_VLForConditionalGeneration (~16.6 GB, 28 layers, streamed)
  - Transformer:  QwenImageTransformer2DModel (60 joint blocks, ~40.9 GB, streamed)
  - VAE:          AutoencoderKLQwenImage (5D video VAE, resident)
  - Scheduler:    FlowMatchEulerDiscreteScheduler

Key Qwen-specific details:
  - guidance_embeds=False, so no guidance conditioning
  - true_cfg_scale: classical CFG, runs transformer TWICE (cond + uncond)
  - Latents: packed spatial latents (in_channels=64 -> 16 real channels + 2x2 packing)
  - VAE decode: outputs 5D tensor (B, C, T, H, W), take frame 0: [:, :, 0]
  - VAE normalization: per-channel latents_mean / latents_std
  - img_shapes: passed to transformer for RoPE
  - Uses mu-shifted sigmas for timestep schedule
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from .transformer_streamer import QwenImageStreamer
from weellm.core.encoders.qwen2_5_vl_streamer import StreamingQwenTextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear interpolation of mu for FlowMatch shift."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * image_seq_len + b


class WeeQwenImagePipeline(BasePipeline):
    """
    Memory-frugal Qwen-Image pipeline.

    Peak VRAM budget:
      VAE (resident) ~0.5 GB
      + Qwen text encoder resident ~1 GB
      + largest Qwen TE layer ~600 MB (streamed)
      + largest transformer block ~700 MB (streamed)
      ≈ ~3 GB total peak
    """

    def __init__(
        self,
        transformer: QwenImageStreamer,
        text_encoder: StreamingQwenTextEncoder,
        vae,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._transformer = transformer
        self._text_encoder = text_encoder
        self._vae = vae
        self._scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self._vae_scale_factor = 8  # 8x spatial compression

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "WeeQwenImagePipeline":
        model_dir = Path(model_dir)

        print("\n============================================================")
        print("  WeeQwenImagePipeline -- initialising")
        print("============================================================\n")

        # 1. Scheduler + Tokenizer
        print("[1/4] Loading scheduler and tokenizer ...")
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2Tokenizer

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_dir / "scheduler")
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir / "tokenizer"))

        # 2. VAE (resident)
        print("\n[2/4] Loading VAE resident on GPU ...")
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(
            str(model_dir / "vae"), torch_dtype=dtype
        ).to(device)
        vae.eval()
        report_memory("After VAE load")

        # 3. Streaming text encoder (Qwen2.5-VL, 28 layers)
        print("\n[3/4] Preparing streaming Qwen text encoder ...")
        text_encoder = StreamingQwenTextEncoder.from_pretrained(
            model_dir=model_dir / "text_encoder",
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
            max_length=512,
        )

        # 4. Streaming transformer (60 joint blocks)
        print("\n[4/4] Preparing streaming Qwen-Image transformer ...")
        transformer = QwenImageStreamer.from_pretrained(
            transformer_dir=model_dir / "transformer",
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        report_memory("After transformer init")

        print("\n============================================================")
        print("  WeeQwenImagePipeline ready.")
        print("============================================================\n")

        return cls(
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Latent packing helpers (mirrors QwenImagePipeline)
    # ------------------------------------------------------------------

    def _pack_latents(self, latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W/p^2, C*p^2)"""
        B, C, H, W = latents.shape
        ph = H // patch_size
        pw = W // patch_size
        latents = latents.view(B, C, ph, patch_size, pw, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, ph * pw, C * patch_size * patch_size)
        return latents

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        patch_size: int = 2,
    ) -> torch.Tensor:
        """(B, seq, C*p^2) -> (B, C, H, W)"""
        B = latents.shape[0]
        ph = height // self._vae_scale_factor // patch_size
        pw = width // self._vae_scale_factor // patch_size
        C = latents.shape[-1] // (patch_size * patch_size)
        latents = latents.view(B, ph, pw, C, patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C, ph * patch_size, pw * patch_size)
        return latents

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = "",
        seed: int = 42,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        do_cfg = true_cfg_scale > 1.0

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps (cfg={true_cfg_scale}) ...")

        # 1. Encode prompt
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds, prompt_embeds_mask = self._text_encoder.encode(prompt)
        if do_cfg:
            neg_embeds, neg_mask = self._text_encoder.encode(negative_prompt)
        report_memory("After text encode")

        # Handle all-ones mask (can be None per diffusers convention)
        if prompt_embeds_mask.all():
            prompt_embeds_mask = None
        if do_cfg and neg_mask.all():
            neg_mask = None

        if do_cfg:
            max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
            
            def pad_seq(emb, mask, tgt_len):
                if emb.shape[1] < tgt_len:
                    pad_len = tgt_len - emb.shape[1]
                    import torch.nn.functional as F
                    emb = F.pad(emb, (0, 0, 0, pad_len))
                    if mask is not None:
                        mask = F.pad(mask, (0, pad_len))
                return emb, mask
                
            if prompt_embeds_mask is None:
                prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=prompt_embeds.dtype, device=prompt_embeds.device)
            if neg_mask is None:
                neg_mask = torch.ones(neg_embeds.shape[:2], dtype=neg_embeds.dtype, device=neg_embeds.device)
                
            prompt_embeds, prompt_embeds_mask = pad_seq(prompt_embeds, prompt_embeds_mask, max_len)
            neg_embeds, neg_mask = pad_seq(neg_embeds, neg_mask, max_len)

            prompt_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
            prompt_embeds_mask = torch.cat([prompt_embeds_mask, neg_mask], dim=0)

        # 2. Prepare latents
        # num_channels_latents = in_channels // 4 = 64 // 4 = 16
        num_channels_latents = self._transformer.model.config.in_channels // 4
        latent_h = height // self._vae_scale_factor
        latent_w = width // self._vae_scale_factor

        latents = torch.randn(
            (1, num_channels_latents, latent_h, latent_w),
            generator=generator, device=self.device, dtype=self.dtype
        )

        # Pack: (B, C, H, W) -> (B, seq, in_channels)
        latents = self._pack_latents(latents, patch_size=2)
        
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
        if do_cfg:
            img_shapes = img_shapes * 2

        # 3. Scheduler with mu-shifted sigmas
        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            base_seq_len=self._scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=self._scheduler.config.get("max_image_seq_len", 4096),
            base_shift=self._scheduler.config.get("base_shift", 0.5),
            max_shift=self._scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self._scheduler.set_timesteps(
            sigmas=sigmas, device=self.device, mu=mu
        )
        timesteps = self._scheduler.timesteps

        # 4. Denoising loop
        print("  [2/3] Denoising ...")
        for i, t in enumerate(timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.1f}) ...")

            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            timestep = t.expand(latent_model_input.shape[0]).to(self.dtype)

            # Single batched pass!
            with self._transformer.cache_context("forward"):
                noise_pred_all = self._transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]

            # True CFG: extract cond/uncond + norm-based blending
            if do_cfg:
                noise_pred, neg_noise_pred = noise_pred_all.chunk(2)
                
                comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cond_norm / comb_norm)
            else:
                noise_pred = noise_pred_all

            latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            report_memory(f"  step {i+1}")

        # 5. Decode
        print("  [3/3] Decoding ...")
        latents = self._unpack_latents(latents, height, width, patch_size=2)
        latents = latents.to(self._vae.dtype)

        # Per-channel denormalization using VAE latents_mean / latents_std
        latents_mean = torch.tensor(self._vae.config.latents_mean).view(
            1, self._vae.config.z_dim, 1, 1
        ).to(device=self.device, dtype=latents.dtype)
        latents_std_inv = 1.0 / torch.tensor(self._vae.config.latents_std).view(
            1, self._vae.config.z_dim, 1, 1
        ).to(device=self.device, dtype=latents.dtype)

        latents = latents / latents_std_inv + latents_mean

        # Add temporal dim for 5D video VAE: (B, C, H, W) -> (B, C, 1, H, W)
        latents = latents.unsqueeze(2)
        
        # Use tiling to prevent massive VRAM spike
        self._vae.enable_tiling()
        image = self._vae.decode(latents, return_dict=False)[0][:, :, 0]
        self._vae.disable_tiling()
        report_memory("After VAE decode")

        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype(np.uint8)

        print("  Done!")
        return Image.fromarray(image)

    __call__ = generate
