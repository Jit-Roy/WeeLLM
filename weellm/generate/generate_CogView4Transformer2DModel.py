import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory


def calculate_shift(image_seq_len, base_seq_len: int = 256, base_shift: float = 0.25, max_shift: float = 0.75):
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


@torch.no_grad()
def generate(
    self,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
    max_sequence_length: int = 1024,
) -> Image.Image:

    generator = torch.Generator(device=self.device).manual_seed(seed)

    print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")
    
    # Generate cond
    prompt_embeds = self._text_encoder.encode(prompt)
    
    # Generate uncond
    negative_prompt = ""
    negative_prompt_embeds = self._text_encoder.encode(negative_prompt)
    
    report_memory("After text encode")
    if hasattr(self, 'free_text_encoder_ram'):
        self.free_text_encoder_ram()

    # CogView4 uses a VAE with scale factor 8
    vae_scale_factor = 8
    latent_channels = self._transformer.model.config.in_channels # usually 16
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=self.device, dtype=torch.float32
    )

    original_size = torch.tensor([[height, width]], dtype=self.dtype, device=self.device)
    target_size = torch.tensor([[height, width]], dtype=self.dtype, device=self.device)
    crops_coords_top_left = torch.tensor([[0, 0]], dtype=self.dtype, device=self.device)

    patch_size = self._transformer.model.config.patch_size
    image_seq_len = (latent_h * latent_w) // (patch_size ** 2)
    
    # Scheduler timesteps
    mu = calculate_shift(
        image_seq_len,
        base_seq_len=self._scheduler.config.get("base_image_seq_len", 256),
        base_shift=self._scheduler.config.get("base_shift", 0.25),
        max_shift=self._scheduler.config.get("max_shift", 0.75),
    )
    
    self._scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
    timesteps = self._scheduler.timesteps

    print("  [2/3] Denoising ...")
    for i, t in enumerate(timesteps):
        print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.4f}) ...")

        latent_model_input = latents.to(self.dtype)
        # Use float32 for timestep to avoid precision issues in sincos embeddings
        timestep = t.expand(1).to(torch.float32)

        with self._transformer.model.cache_context("cond"):
            noise_pred_cond = self._transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                original_size=original_size,
                target_size=target_size,
                crop_coords=crops_coords_top_left,
                return_dict=False,
            )[0]

        with self._transformer.model.cache_context("uncond"):
            noise_pred_uncond = self._transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=negative_prompt_embeds,
                timestep=timestep,
                original_size=original_size,
                target_size=target_size,
                crop_coords=crops_coords_top_left,
                return_dict=False,
            )[0]

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        report_memory(f"  step {i+1}")

    print("  [3/3] Decoding ...")
    latents = latents / self._vae.config.scaling_factor
    latents = latents.to(dtype=self.dtype)
    image = self._vae.decode(latents, return_dict=False)[0]
    report_memory("After VAE decode")

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)

    print("  Done!")
    return Image.fromarray(image)
