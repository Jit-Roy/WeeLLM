"""
pipeline.py -- WeeFluxSchnellPipeline

Memory-frugal inference pipeline for FLUX.1-schnell (and FLUX.1-dev).

Architecture vs FLUX.2-klein:
  - Text encoder 1: CLIPTextModel (~250 MB, kept RESIDENT — small enough)
  - Text encoder 2: T5EncoderModel-XXL (~9 GB, streamed layer-by-layer)
  - Transformer:    FluxTransformer2DModel (19 double + 38 single blocks, streamed)
  - VAE:            AutoencoderKL (standard, ~168 MB, resident)
  - Scheduler:      FlowMatchEulerDiscreteScheduler

Key differences from FLUX.2-klein:
  - guidance_embeds=False in config (no guidance conditioning)
  - Latent packing: same 2×2 spatial pack, but in_channels=64 (not 128)
    => num_latent_channels = 16 (not 32)
  - T5 prompt embeds: (1, seq_len, 4096)
  - CLIP pooled embeds: (1, 768)  [used as pooled_projections input]
  - Standard VAE (no BN denorm, no _unpatchify_latents needed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .transformer_streamer import FluxSchnellStreamer
from weellm.core.encoders.t5_streamer import StreamingT5Encoder
from weellm.core.encoders.clip_streamer import StreamingCLIPTextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


class WeeFluxSchnellPipeline(BasePipeline):
    """
    Memory-frugal FLUX.1-schnell pipeline.

    Peak VRAM budget:
      CLIP (resident) ~250 MB
      + VAE (resident) ~168 MB
      + largest T5 block ~500 MB (streamed)
      + largest Flux double block ~600 MB (streamed)
      ≈ ~1.5 GB total peak
    """

    def __init__(
        self,
        transformer: FluxSchnellStreamer,
        text_encoder: StreamingCLIPTextEncoder,   # CLIP (resident)
        text_encoder_2: StreamingT5Encoder,        # T5 (streamed)
        vae,
        tokenizer,
        tokenizer_2,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._transformer = transformer
        self._text_encoder = text_encoder
        self._text_encoder_2 = text_encoder_2
        self._vae = vae
        self._tokenizer = tokenizer
        self._tokenizer_2 = tokenizer_2
        self._scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self._vae_scale_factor = 16   # FLUX.1 VAE: 16x spatial compression

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "WeeFluxSchnellPipeline":
        model_dir = Path(model_dir)

        print("\n============================================================")
        print("  WeeFluxSchnellPipeline -- initialising")
        print("============================================================\n")

        # 1. Scheduler + Tokenizers
        print("[1/5] Loading scheduler and tokenizers ...")
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTokenizer, T5TokenizerFast

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_dir / "scheduler")
        )
        tokenizer = CLIPTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        tokenizer_2 = T5TokenizerFast.from_pretrained(str(model_dir / "tokenizer_2"))

        # 2. VAE (resident — AutoencoderKL, ~168 MB)
        print("\n[2/5] Loading VAE resident on GPU ...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            str(model_dir / "vae"), torch_dtype=dtype
        ).to(device)
        vae.eval()
        report_memory("After VAE load")

        # 3. CLIP text encoder (resident — ~250 MB, small enough to keep in VRAM)
        print("\n[3/5] Loading CLIP text encoder (resident on GPU) ...")
        from transformers import CLIPTextModel
        clip_encoder = CLIPTextModel.from_pretrained(
            str(model_dir / "text_encoder"), torch_dtype=dtype
        ).to(device)
        clip_encoder.eval()
        # Wrap in a thin callable that matches StreamingCLIPTextEncoder interface
        # (We load it fully resident since it's only ~250 MB)
        report_memory("After CLIP load")

        # 4. T5 text encoder (streamed — 9 GB, 24 blocks)
        print("\n[4/5] Preparing streaming T5 text encoder ...")
        t5_encoder = StreamingT5Encoder.from_pretrained(
            model_dir=str(model_dir / "text_encoder_2"),
            device=device,
            dtype=dtype,
            max_length=256,
        )
        report_memory("After T5 encoder init")

        # 5. Transformer (streamed — 19 double + 38 single blocks)
        print("\n[5/5] Preparing streaming Flux transformer ...")
        transformer = FluxSchnellStreamer.from_pretrained(
            transformer_dir=model_dir / "transformer",
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        report_memory("After transformer init")

        print("\n============================================================")
        print("  WeeFluxSchnellPipeline ready.")
        print("============================================================\n")

        return cls(
            transformer=transformer,
            text_encoder=clip_encoder,
            text_encoder_2=t5_encoder,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Helpers: latent packing (same 2×2 spatial pack as FLUX.2-klein)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
        """
        Pack spatial latents into sequence tokens.
        (B, C, H, W) -> (B, H//p * W//p, C * p * p)
        FLUX.1 packs with patch_size=2:
          (1, 16, H//8, W//8) -> (1, H//16 * W//16, 64)
        """
        B, C, H, W = latents.shape
        p = patch_size
        latents = latents.reshape(B, C, H // p, p, W // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        latents = latents.reshape(B, (H // p) * (W // p), C * p * p)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, patch_size: int = 2) -> torch.Tensor:
        """
        Unpack sequence tokens back into spatial latents.
        (B, seq_len, C*p*p) -> (B, C, H, W)
        """
        B, seq_len, Cpp = latents.shape
        p = patch_size
        C = Cpp // (p * p)
        h = height // p
        w = width // p
        latents = latents.reshape(B, h, w, C, p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        latents = latents.reshape(B, C, h * p, w * p)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Prepare image position IDs for the RoPE embedder (patch-level)."""
        h = height // 2   # after packing with patch_size=2
        w = width // 2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]
        return ids.reshape(1, h * w, 3).expand(1, -1, -1)  # (1, h*w, 3)

    @staticmethod
    def _prepare_text_ids(seq_len: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Prepare text position IDs (all zeros for FLUX.1)."""
        return torch.zeros(1, seq_len, 3, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, max_t5_length: int = 256):
        """
        Returns:
          prompt_embeds:        (1, seq_len, 4096) from T5
          pooled_prompt_embeds: (1, 768)           from CLIP
        """
        # --- CLIP pooled embed ---
        clip_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            clip_out = self._text_encoder(
                clip_inputs.input_ids,
                output_hidden_states=False,
                return_dict=False,
            )
        # CLIPTextModel return_dict=False: (last_hidden_state, pooler_output)
        pooled_prompt_embeds = clip_out[1].to(dtype=self.dtype)  # (1, 768)
        del clip_inputs, clip_out

        # --- T5 sequence embed ---
        t5_inputs = self._tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_t5_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        t5_out = self._text_encoder_2(t5_inputs.input_ids)
        # T5EncoderModel return_dict=False: (last_hidden_state,)
        prompt_embeds = t5_out[0].to(dtype=self.dtype)  # (1, seq_len, 4096)
        del t5_inputs, t5_out
        clean_memory(self.device)

        return prompt_embeds, pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,    # schnell is distilled, no CFG needed
        seed: int = 42,
        max_t5_length: int = 256,
    ) -> Image.Image:

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

        # 1. Encode prompt
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt, max_t5_length)
        report_memory("After text encode")

        # 2. Prepare latents
        # FLUX.1 VAE: 8x compression. After 2x2 patch packing: effective 16x.
        # Latent shape before packing: (B, 16, H//8, W//8)
        latent_h = height // 8
        latent_w = width // 8
        latents = torch.randn(
            (1, 16, latent_h, latent_w),
            generator=generator, device=self.device, dtype=self.dtype
        )

        # Pack latents: (1, 16, H//8, W//8) -> (1, seq_len, 64)
        packed_latents = self._pack_latents(latents, patch_size=2)
        seq_len = packed_latents.shape[1]  # (H//16) * (W//16)

        # Prepare position IDs
        latent_image_ids = self._prepare_latent_image_ids(latent_h, latent_w, self.device, self.dtype)
        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], self.device, self.dtype)

        # Scheduler
        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self._scheduler.timesteps

        # 3. Denoising loop
        print("  [2/3] Denoising ...")
        for i, t in enumerate(timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.4f}) ...")

            latent_model_input = packed_latents.to(self.dtype)
            timestep = t.expand(1).to(self.dtype)

            noise_pred = self._transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            packed_latents = self._scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
            report_memory(f"  step {i+1}")

        # 4. Unpack + decode
        print("  [3/3] Decoding ...")
        latents = self._unpack_latents(packed_latents, latent_h, latent_w, patch_size=2)
        # latents: (1, 16, H//8, W//8)

        latents = (latents / self._vae.config.scaling_factor) + self._vae.config.shift_factor
        image = self._vae.decode(latents, return_dict=False)[0]
        report_memory("After VAE decode")

        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype(np.uint8)

        print("  Done!")
        return Image.fromarray(image)
    __call__ = generate
