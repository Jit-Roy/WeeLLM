import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@torch.no_grad()
def generate(
    self,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    seed: int = 42,
    max_sequence_length: int = 256,
) -> Image.Image:

    generator = torch.Generator(device=self.device).manual_seed(seed)

    print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")
    
    system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts."
    formatted_prompt = f"{system_prompt} <Prompt Start> {prompt}"

    # Generate cond
    prompt_embeds = self._text_encoder.encode(formatted_prompt)
    prompt_attention_mask = torch.ones(
        (1, prompt_embeds.shape[1]), dtype=torch.bool, device=self.device
    )

    # Generate uncond
    formatted_negative = f"{system_prompt} <Prompt Start> "
    negative_prompt_embeds = self._text_encoder.encode(formatted_negative)
    negative_prompt_attention_mask = torch.ones(
        (1, negative_prompt_embeds.shape[1]), dtype=torch.bool, device=self.device
    )
    
    report_memory("After text encode")
    if hasattr(self, 'free_text_encoder_ram'):
        self.free_text_encoder_ram()

    # Lumina uses VAE scale factor 8, but also requires divisible by 16 because of patching
    vae_scale_factor = 8
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latent_channels = self._transformer.model.config.in_channels
    latent_h = height
    latent_w = width

    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=self.device, dtype=torch.float32
    )

    patch_size = self._transformer.model.config.patch_size
    image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
    
    mu = calculate_shift(
        image_seq_len,
        base_seq_len=self._scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=self._scheduler.config.get("max_image_seq_len", 4096),
        base_shift=self._scheduler.config.get("base_shift", 0.5),
        max_shift=self._scheduler.config.get("max_shift", 1.15),
    )
    
    self._scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
    timesteps = self._scheduler.timesteps
    
    cfg_trunc_ratio = 1.0
    cfg_normalization = True

    print("  [2/3] Denoising ...")
    for i, t in enumerate(timesteps):
        print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.4f}) ...")

        latent_model_input = latents.to(self.dtype)
        
        do_classifier_free_truncation = (i + 1) / num_inference_steps > cfg_trunc_ratio
        current_timestep = 1 - (t / self._scheduler.config.num_train_timesteps)
        current_timestep = current_timestep.expand(latents.shape[0]).to(torch.float32)

        noise_pred_cond = self._transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0 and not do_classifier_free_truncation:
            noise_pred_uncond = self._transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=negative_prompt_attention_mask,
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            if cfg_normalization:
                cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_pred = noise_pred * (cond_norm / noise_norm)
        else:
            noise_pred = noise_pred_cond

        noise_pred = -noise_pred
        latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        report_memory(f"  step {i+1}")

    print("  [3/3] Decoding ...")
    
    shift_factor = getattr(self._vae.config, "shift_factor", 0.0)
    latents = (latents / self._vae.config.scaling_factor) + shift_factor
    latents = latents.to(dtype=self.dtype)
    
    image = self._vae.decode(latents, return_dict=False)[0]
    report_memory("After VAE decode")

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)

    print("  Done!")
    return Image.fromarray(image)
