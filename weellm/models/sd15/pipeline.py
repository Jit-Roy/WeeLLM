"""
pipeline.py -- WeeLLM streaming pipeline for Stable Diffusion v1.x / v2.x.

Architecture differences vs SDXL:
  - Single CLIPTextModel (not two encoders, no projection head)
  - No pooled embeddings, no add_time_ids for the UNet
  - Native resolution: 512x512
  - Scheduler: PNDMScheduler (can also use EulerDiscreteScheduler)
  - UNet takes only: latent, timestep, encoder_hidden_states
"""

import os
from typing import Union
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from weellm.models.sdxl.unet_streamer import UNetStreamer
from weellm.core.encoders.clip_streamer import StreamingCLIPTextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


class WeeSD15Pipeline(BasePipeline):
    """
    Streaming pipeline for Stable Diffusion v1.x / v2.x.
    Streams one CLIP text encoder and UNet block-by-block to stay under 4 GB VRAM.
    """

    def __init__(
        self,
        unet: UNetStreamer,
        text_encoder: StreamingCLIPTextEncoder,
        vae: AutoencoderKL,
        tokenizer: CLIPTokenizer,
        scheduler: PNDMScheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._unet = unet
        self._text_encoder = text_encoder
        self._vae = vae
        self._tokenizer = tokenizer
        self._scheduler = scheduler
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs
    ):
        model_dir = str(model_dir)
        print("\n============================================================")
        print("  WeeSD15Pipeline -- initialising")
        print("============================================================\n")

        # 1. Scheduler and Tokenizer
        print("[1/4] Loading scheduler and tokenizer ...")
        scheduler = PNDMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

        # 2. VAE resident on GPU (small, ~83 MB for SD1.5)
        print("\n[2/4] Loading VAE resident on GPU ...")
        try:
            vae = AutoencoderKL.from_pretrained(
                model_dir, subfolder="vae", torch_dtype=dtype, variant="fp16", use_safetensors=True
            ).to(device)
        except OSError:
            vae = AutoencoderKL.from_pretrained(
                model_dir, subfolder="vae", torch_dtype=dtype, use_safetensors=True
            ).to(device)
        report_memory("After VAE load")

        # 3. Text Encoder (streaming)
        print("\n[3/4] Preparing streaming Text Encoder ...")
        # SD1.5 only uses the last hidden state (no hidden_states pool), but we still
        # need output_hidden_states=False here since we grab te_out[0] directly.
        text_encoder = StreamingCLIPTextEncoder.from_pretrained(
            CLIPTextModel, model_dir, "text_encoder",
            device=device, dtype=dtype, output_hidden_states=False
        )
        report_memory("After text encoder init")

        # 4. UNet (streaming) -- reuses the same UNetStreamer as SDXL
        print("\n[4/4] Preparing streaming UNet ...")
        unet = UNetStreamer.from_pretrained(model_dir, device, dtype, prefetch)
        report_memory("After unet init")

        print("\n============================================================")
        print("  SD1.5 Pipeline ready.")
        print("============================================================\n")

        return cls(
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        """
        Encodes a prompt into CLIP embeddings.
        SD1.5 UNet expects: encoder_hidden_states shape [B, 77, 768]
        We use the last hidden state (te_out[0]) directly.
        """
        text_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # return_dict=False -> (last_hidden_state, pooler_output)
        te_out = self._text_encoder(text_inputs.input_ids)
        prompt_embeds = te_out[0]   # [1, 77, 768]

        del text_inputs, te_out
        clean_memory(self.device)

        return prompt_embeds

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

        # 1. Encode prompt
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds = self._encode_prompt(prompt)

        if guidance_scale > 1.0:
            uncond_embeds = self._encode_prompt("")
            # Classifier-free guidance: batch [uncond, cond]
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            del uncond_embeds

        report_memory("After text encode")

        # 2. Prepare latents
        shape = (1, self._unet.model.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)

        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self._scheduler.init_noise_sigma

        # 3. Denoising loop
        print("  [2/3] Denoising ...")
        for i, t in enumerate(self._scheduler.timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.1f}) ...")

            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self._scheduler.scale_model_input(latent_model_input, t)

            # SD1.5 UNet: no added_cond_kwargs (unlike SDXL)
            noise_pred = self._unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self._scheduler.step(noise_pred, t, latents).prev_sample
            report_memory(f"  step {i+1}")

        # 4. Decode latents
        print("  [3/3] Decoding ...")
        needs_upcasting = self._vae.dtype == torch.float16 and self._vae.config.force_upcast
        if needs_upcasting:
            self._vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)

        latents = latents / self._vae.config.scaling_factor
        image = self._vae.decode(latents).sample

        if needs_upcasting:
            self._vae.to(dtype=self.dtype)

        report_memory("After VAE decode")

        # 5. Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype(np.uint8)

        print("  Done!")
        return Image.fromarray(image)
