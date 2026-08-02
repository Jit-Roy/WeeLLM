"""
pipeline.py -- WeeSD35Pipeline

Memory-frugal inference pipeline for Stable Diffusion 3.5 Medium.

Architecture:
  - Text encoder 1: CLIPTextModelWithProjection (~247 MB, resident on GPU)
  - Text encoder 2: CLIPTextModelWithProjection (~1.38 GB, resident on GPU)
  - Text encoder 3: T5EncoderModel-XXL (~9.5 GB, streamed layer-by-layer)
  - Transformer:    SD3Transformer2DModel (24 joint blocks, ~4.9 GB, streamed)
  - VAE:            AutoencoderKL (resident, ~168 MB)
  - Scheduler:      FlowMatchEulerDiscreteScheduler

Key differences from FLUX:
  - Uses 3 text encoders (CLIP x2 + T5)
  - Pooled embeddings = concat of CLIP1 + CLIP2 pooled (2048 dim)
  - Sequence embeddings = T5 hidden states (4096 dim)
  - Latent channels = 16 (same as FLUX.1)
  - No latent packing (SD3 uses standard spatial latents)
  - Standard SD3 CFG guidance (guidance_scale typically 4.5-7.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from .transformer_streamer import SD35Streamer
from weellm.core.encoders.t5_streamer import StreamingT5Encoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


class WeeSD35Pipeline(BasePipeline):
    """
    Memory-frugal SD 3.5 Medium pipeline.

    Peak VRAM budget:
      CLIP-1 (resident) ~247 MB
      + CLIP-2 (resident) ~1.38 GB
      + VAE (resident) ~168 MB
      + largest T5 block ~500 MB (streamed)
      + largest SD3 joint block ~300 MB (streamed)
      ≈ ~2.5 GB total peak
    """

    def __init__(
        self,
        transformer: SD35Streamer,
        text_encoder,           # CLIPTextModelWithProjection (CLIP-1, resident)
        text_encoder_2,         # CLIPTextModelWithProjection (CLIP-2, resident)
        text_encoder_3: StreamingT5Encoder,  # T5-XXL (streamed)
        vae,
        tokenizer,
        tokenizer_2,
        tokenizer_3,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._transformer = transformer
        self._text_encoder = text_encoder
        self._text_encoder_2 = text_encoder_2
        self._text_encoder_3 = text_encoder_3
        self._vae = vae
        self._tokenizer = tokenizer
        self._tokenizer_2 = tokenizer_2
        self._tokenizer_3 = tokenizer_3
        self._scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self._vae_scale_factor = 8  # SD3.5 VAE: 8x spatial compression

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "WeeSD35Pipeline":
        model_dir = Path(model_dir)

        print("\n============================================================")
        print("  WeeSD35Pipeline -- initialising")
        print("============================================================\n")

        # 1. Scheduler + Tokenizers
        print("[1/6] Loading scheduler and tokenizers ...")
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTokenizer, T5TokenizerFast

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_dir / "scheduler")
        )
        tokenizer = CLIPTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(model_dir / "tokenizer_2"))
        tokenizer_3 = T5TokenizerFast.from_pretrained(str(model_dir / "tokenizer_3"))

        # 2. VAE (resident)
        print("\n[2/6] Loading VAE resident on GPU ...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            str(model_dir / "vae"), torch_dtype=dtype
        ).to(device)
        vae.eval()
        report_memory("After VAE load")

        # 3. CLIP-1 text encoder (~247 MB, resident)
        print("\n[3/6] Loading CLIP-1 text encoder (resident on GPU) ...")
        from transformers import CLIPTextModelWithProjection
        clip1 = CLIPTextModelWithProjection.from_pretrained(
            str(model_dir / "text_encoder"), torch_dtype=dtype
        ).to(device)
        clip1.eval()
        report_memory("After CLIP-1 load")

        # 4. CLIP-2 text encoder (~1.38 GB, resident)
        print("\n[4/6] Loading CLIP-2 text encoder (resident on GPU) ...")
        clip2 = CLIPTextModelWithProjection.from_pretrained(
            str(model_dir / "text_encoder_2"), torch_dtype=dtype
        ).to(device)
        clip2.eval()
        report_memory("After CLIP-2 load")

        # 5. T5 text encoder (streamed)
        print("\n[5/6] Preparing streaming T5 text encoder ...")
        t5_encoder = StreamingT5Encoder.from_pretrained(
            model_dir=str(model_dir / "text_encoder_3"),
            device=device,
            dtype=dtype,
            max_length=256,
        )
        report_memory("After T5 encoder init")

        # 6. Transformer (streamed — 24 joint blocks)
        print("\n[6/6] Preparing streaming SD3.5 transformer ...")
        transformer = SD35Streamer.from_pretrained(
            transformer_dir=model_dir / "transformer",
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        report_memory("After transformer init")

        print("\n============================================================")
        print("  WeeSD35Pipeline ready.")
        print("============================================================\n")

        return cls(
            transformer=transformer,
            text_encoder=clip1,
            text_encoder_2=clip2,
            text_encoder_3=t5_encoder,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, max_t5_length: int = 256):
        """
        Returns:
          prompt_embeds:        (1, seq_len, 4096) from T5
          pooled_prompt_embeds: (1, 2048)           from concat(CLIP-1, CLIP-2)
        """
        # --- CLIP-1 ---
        clip1_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            clip1_out = self._text_encoder(
                clip1_inputs.input_ids,
                output_hidden_states=True,
            )
        # Use second-to-last hidden state (SD3 convention)
        clip1_hidden = clip1_out.hidden_states[-2]  # (1, 77, 768)
        clip1_pooled = clip1_out.text_embeds         # (1, 768)
        del clip1_inputs, clip1_out

        # --- CLIP-2 ---
        clip2_inputs = self._tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self._tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            clip2_out = self._text_encoder_2(
                clip2_inputs.input_ids,
                output_hidden_states=True,
            )
        clip2_hidden = clip2_out.hidden_states[-2]  # (1, 77, 1280)
        clip2_pooled = clip2_out.text_embeds         # (1, 1280)
        del clip2_inputs, clip2_out

        # Concat CLIP hidden states for the context: (1, 77, 2048)
        clip_combined = torch.cat([clip1_hidden, clip2_hidden], dim=-1)

        # Pooled = concat of both CLIP pooled: (1, 2048)
        pooled_prompt_embeds = torch.cat([clip1_pooled, clip2_pooled], dim=-1).to(dtype=self.dtype)
        del clip1_hidden, clip2_hidden, clip1_pooled, clip2_pooled

        # --- T5 sequence embed ---
        t5_inputs = self._tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_t5_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        t5_out = self._text_encoder_3(t5_inputs.input_ids)
        t5_hidden = t5_out[0].to(dtype=self.dtype)  # (1, seq_len, 4096)
        del t5_inputs, t5_out

        # SD3 prompt embed = concat of padded CLIP + T5 along sequence dim
        # Pad clip_combined to match T5 hidden dim (4096)
        clip_padded = torch.nn.functional.pad(
            clip_combined, (0, 4096 - clip_combined.shape[-1])
        ).to(dtype=self.dtype)  # (1, 77, 4096)

        # prompt_embeds = concat along seq dim: (1, 77 + max_t5_length, 4096)
        prompt_embeds = torch.cat([clip_padded, t5_hidden], dim=1)
        del clip_combined, clip_padded, t5_hidden
        clean_memory(self.device)

        return prompt_embeds, pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        seed: int = 42,
        max_t5_length: int = 256,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps (guidance={guidance_scale}) ...")

        # 1. Encode prompt
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt, max_t5_length)

        # For CFG we need negative embeds. Use empty prompt.
        negative_prompt_embeds, negative_pooled = self._encode_prompt("", max_t5_length)
        report_memory("After text encode")

        # 2. Prepare latents
        # SD3.5 VAE: 8x spatial compression, 16 latent channels
        latent_h = height // self._vae_scale_factor
        latent_w = width // self._vae_scale_factor
        latents = torch.randn(
            (1, 16, latent_h, latent_w),
            generator=generator, device=self.device, dtype=self.dtype
        )

        # Scheduler
        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self._scheduler.timesteps

        # 3. Denoising loop
        print("  [2/3] Denoising ...")
        for i, t in enumerate(timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.1f}) ...")

            # CFG: run both conditional and unconditional
            latent_model_input = torch.cat([latents, latents])
            timestep = t.expand(2).long()   # SD3 expects LongTensor timesteps
            enc_hidden = torch.cat([negative_prompt_embeds, prompt_embeds])
            enc_pooled = torch.cat([negative_pooled, pooled_prompt_embeds])

            noise_pred = self._transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                pooled_projections=enc_pooled,
                return_dict=False,
            )[0]

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            report_memory(f"  step {i+1}")

        # 4. Decode
        print("  [3/3] Decoding ...")
        # SD3 VAE: decode with shift_factor and scaling_factor
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
