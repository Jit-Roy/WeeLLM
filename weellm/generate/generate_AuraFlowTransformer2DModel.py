import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

@torch.no_grad()
def _auraflow_encode_prompt(self, prompt: str, max_t5_length: int = 256):
    t5_inputs = self._tokenizer(
        prompt,
        padding="max_length",
        max_length=max_t5_length,
        truncation=True,
        return_tensors="pt",
    ).to(self.device)

    t5_out = self._text_encoder(t5_inputs.input_ids)
    prompt_embeds = t5_out[0].to(dtype=self.dtype)
    
    attention_mask = t5_inputs.attention_mask.unsqueeze(-1).expand(prompt_embeds.shape).to(self.dtype)
    prompt_embeds = prompt_embeds * attention_mask
    
    del t5_inputs, t5_out
    clean_memory(self.device)

    return prompt_embeds

@torch.no_grad()
def generate(
    self,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int = 42,
    max_t5_length: int = 256,
) -> Image.Image:

    generator = torch.Generator(device=self.device).manual_seed(seed)

    print(f"\\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")
    prompt_embeds = _auraflow_encode_prompt(self, prompt, max_t5_length)
    
    # Conditional + Unconditional logic for classifier-free guidance
    uncond_embeds = _auraflow_encode_prompt(self, "", max_t5_length)
    
    report_memory("After text encode")
    if hasattr(self, 'free_text_encoder_ram'):
        self.free_text_encoder_ram()

    # VAE Scale factor for AutoencoderKL is typically 8. 
    # AuraFlow uses 4 latent channels.
    latent_h = height // 8
    latent_w = width // 8
    
    latents = torch.randn(
        (1, 4, latent_h, latent_w),
        generator=generator, device=self.device, dtype=self.dtype
    )

    self._scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps = self._scheduler.timesteps

    print("  [2/3] Denoising ...")
    for i, t in enumerate(timesteps):
        print(f"        Step {i+1}/{num_inference_steps} (t={t.item():.4f}) ...")

        # Duplicate inputs for classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0).to(self.dtype)
        timestep = t.expand(2).to(self.dtype)
        encoder_hidden_states = torch.cat([uncond_embeds, prompt_embeds], dim=0).to(self.dtype)

        noise_pred = self._transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000.0,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        report_memory(f"  step {i+1}")

    print("  [3/3] Decoding ...")
    
    # Unscale latents
    shift = self._vae.config.shift_factor if getattr(self._vae.config, "shift_factor", None) is not None else 0.0
    latents = (latents / self._vae.config.scaling_factor) + shift
    latents = latents.to(dtype=self.dtype)
    image = self._vae.decode(latents, return_dict=False)[0]
    report_memory("After VAE decode")

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)

    print("  Done!")
    return Image.fromarray(image)
