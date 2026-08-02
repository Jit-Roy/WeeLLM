import os
from typing import Optional, Union, List
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .unet_streamer import UNetStreamer
from weellm.core.encoders.clip_streamer import StreamingCLIPTextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory

class WeeSDXLPipeline(BasePipeline):
    """
    Streaming pipeline for SDXL-based architectures.
    Streams two CLIP text encoders and one UNet to stay well under 4 GB VRAM.
    """
    def __init__(
        self,
        unet: UNetStreamer,
        text_encoder_1: StreamingCLIPTextEncoder,
        text_encoder_2: StreamingCLIPTextEncoder,
        vae: AutoencoderKL,
        tokenizer_1: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        scheduler: EulerDiscreteScheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._unet = unet
        self._text_encoder_1 = text_encoder_1
        self._text_encoder_2 = text_encoder_2
        self._vae = vae
        self._tokenizer_1 = tokenizer_1
        self._tokenizer_2 = tokenizer_2
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
        print("  WeeSDXLPipeline -- initialising")
        print("============================================================\n")

        # 1. Load Scheduler and Tokenizers
        print("[1/4] Loading scheduler and tokenizers ...")
        scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        tokenizer_1 = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer_2")

        # 2. Load VAE to GPU (Resident)
        print(f"\n[2/4] Loading VAE resident on GPU ...")
        try:
            vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype, variant="fp16", use_safetensors=True).to(device)
        except OSError:
            vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
        report_memory("After VAE load")

        # 3. Load Text Encoders (Streaming)
        print("\n[3/4] Preparing streaming Text Encoders ...")
        text_encoder_1 = StreamingCLIPTextEncoder.from_pretrained(
            CLIPTextModel, model_dir, "text_encoder",
            device=device, dtype=dtype, output_hidden_states=True
        )
        text_encoder_2 = StreamingCLIPTextEncoder.from_pretrained(
            CLIPTextModelWithProjection, model_dir, "text_encoder_2",
            device=device, dtype=dtype, output_hidden_states=True
        )
        report_memory("After text encoders init")

        # 4. Load UNet (Streaming)
        print("\n[4/4] Preparing streaming UNet ...")
        unet = UNetStreamer.from_pretrained(model_dir, device, dtype, prefetch)
        report_memory("After unet init")

        print("\n============================================================")
        print("  SDXL Pipeline ready.")
        print("============================================================\n")

        return cls(
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        # Text Encoder 1  -- returns tuple when return_dict=False
        text_inputs_1 = self._tokenizer_1(
            prompt, padding="max_length", max_length=self._tokenizer_1.model_max_length, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        # Tuple: (last_hidden_state, pooler_output, *hidden_states)
        te1_out = self._text_encoder_1(text_inputs_1.input_ids)
        # hidden_states is the 3rd element onward; [-2] = second-to-last
        prompt_embeds_1 = te1_out[2][-2]   # te1_out[2] = all_hidden_states tuple

        # Text Encoder 2
        text_inputs_2 = self._tokenizer_2(
            prompt, padding="max_length", max_length=self._tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        # CLIPTextModelWithProjection returns (text_embeds, last_hidden_state, *hidden_states)
        te2_out = self._text_encoder_2(text_inputs_2.input_ids)
        pooled_prompt_embeds = te2_out[0]   # text_embeds (pooled)
        prompt_embeds_2 = te2_out[2][-2]    # second-to-last hidden state

        # Concat embeddings along feature dim
        prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        del text_inputs_1, text_inputs_2, te1_out, te2_out, prompt_embeds_1, prompt_embeds_2
        clean_memory(self.device)
        
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")
        
        # 1. Encode prompt
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)
        report_memory("After text encode")
        
        if guidance_scale > 1.0:
            # Generate unconditioned embeddings
            uncond_embeds, uncond_pooled = self._encode_prompt("")
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)

        # 2. Prepare latents
        shape = (1, self._unet.model.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        
        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self._scheduler.init_noise_sigma

        # SDXL Specific time ids (original_size, crops_coords_top_left, target_size)
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=self.dtype, device=self.device)
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        # 3. Denoising loop
        print("  [2/3] Denoising ...")
        for i, t in enumerate(self._scheduler.timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t.item()}) ...")
            
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self._scheduler.scale_model_input(latent_model_input, t)

            # UNet forward pass (this streams the down/mid/up blocks)
            noise_pred = self._unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample
            
            report_memory(f"  step {i+1}")

        # 4. Decode
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
        
        # 5. Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype(np.uint8)
        
        print("  Done!")
        return Image.fromarray(image)
