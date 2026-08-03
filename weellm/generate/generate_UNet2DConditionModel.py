import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

@torch.no_grad()
def _sd_encode_prompt(self, prompt: str):
    """
        Encodes a prompt into CLIP embeddings.
        SD1.5 UNet expects: encoder_hidden_states shape [B, 77, 768]
        We use the last hidden state (te_out[0]) directly.
        """
    text_inputs = self._tokenizer(prompt, padding='max_length', max_length=self._tokenizer.model_max_length, truncation=True, return_tensors='pt').to(self.device)
    te_out = self._text_encoder(text_inputs.input_ids)
    prompt_embeds = te_out[0]
    del text_inputs, te_out
    clean_memory(self.device)
    return prompt_embeds

@torch.no_grad()
def _sdxl_encode_prompt(self, prompt: str):
    text_inputs_1 = self._tokenizer(prompt, padding='max_length', max_length=self._tokenizer.model_max_length, truncation=True, return_tensors='pt').to(self.device)
    te1_out = self._text_encoder(text_inputs_1.input_ids)
    prompt_embeds_1 = te1_out[2][-2]
    text_inputs_2 = self._tokenizer_2(prompt, padding='max_length', max_length=self._tokenizer_2.model_max_length, truncation=True, return_tensors='pt').to(self.device)
    te2_out = self._text_encoder_2(text_inputs_2.input_ids)
    pooled_prompt_embeds = te2_out[0]
    prompt_embeds_2 = te2_out[2][-2]
    prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    del text_inputs_1, text_inputs_2, te1_out, te2_out, prompt_embeds_1, prompt_embeds_2
    clean_memory(self.device)
    return (prompt_embeds, pooled_prompt_embeds)

def _generate_sd(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=20, guidance_scale: float=7.5, seed: int=42) -> Image.Image:
    generator = torch.Generator(device=self.device).manual_seed(seed)
    print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps ...')
    print('  [1/3] Encoding prompt ...')
    report_memory('Before text encode')
    prompt_embeds = _sd_encode_prompt(self, prompt)
    if guidance_scale > 1.0:
        uncond_embeds = _sd_encode_prompt(self, '')
        prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        del uncond_embeds
    report_memory('After text encode')
    shape = (1, self._unet.model.config.in_channels, height // 8, width // 8)
    latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
    self._scheduler.set_timesteps(num_inference_steps, device=self.device)
    latents = latents * self._scheduler.init_noise_sigma
    print('  [2/3] Denoising ...')
    for i, t in enumerate(self._scheduler.timesteps):
        print(f'        Step {i + 1}/{num_inference_steps} (t={t.item():.1f}) ...')
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = self._scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self._unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self._scheduler.step(noise_pred, t, latents).prev_sample
        report_memory(f'  step {i + 1}')
    print('  [3/3] Decoding ...')
    needs_upcasting = self._vae.dtype == torch.float16 and self._vae.config.force_upcast
    if needs_upcasting:
        self._vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
    latents = latents / self._vae.config.scaling_factor
    image = self._vae.decode(latents).sample
    if needs_upcasting:
        self._vae.to(dtype=self.dtype)
    report_memory('After VAE decode')
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    print('  Done!')
    return Image.fromarray(image)

def _generate_sdxl(self, prompt: str, height: int=1024, width: int=1024, num_inference_steps: int=20, guidance_scale: float=5.0, seed: int=42) -> Image.Image:
    generator = torch.Generator(device=self.device).manual_seed(seed)
    print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps ...')
    print('  [1/3] Encoding prompt ...')
    report_memory('Before text encode')
    prompt_embeds, pooled_prompt_embeds = _sdxl_encode_prompt(self, prompt)
    report_memory('After text encode')
    if guidance_scale > 1.0:
        uncond_embeds, uncond_pooled = _sdxl_encode_prompt(self, '')
        prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)
    shape = (1, self._unet.model.config.in_channels, height // 8, width // 8)
    latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
    self._scheduler.set_timesteps(num_inference_steps, device=self.device)
    latents = latents * self._scheduler.init_noise_sigma
    add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=self.dtype, device=self.device)
    if guidance_scale > 1.0:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    added_cond_kwargs = {'text_embeds': pooled_prompt_embeds, 'time_ids': add_time_ids}
    print('  [2/3] Denoising ...')
    for i, t in enumerate(self._scheduler.timesteps):
        print(f'        Step {i + 1}/{num_inference_steps} (t={t.item()}) ...')
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = self._scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self._unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self._scheduler.step(noise_pred, t, latents).prev_sample
        report_memory(f'  step {i + 1}')
    print('  [3/3] Decoding ...')
    needs_upcasting = self._vae.dtype == torch.float16 and self._vae.config.force_upcast
    if needs_upcasting:
        self._vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
    latents = latents / self._vae.config.scaling_factor
    image = self._vae.decode(latents).sample
    if needs_upcasting:
        self._vae.to(dtype=self.dtype)
    report_memory('After VAE decode')
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    print('  Done!')
    return Image.fromarray(image)


@torch.no_grad()
def generate(self, prompt: str, **kwargs):
    if "tokenizer_2" in self.tokenizers:
        return _generate_sdxl(self, prompt, **kwargs)
    else:
        return _generate_sd(self, prompt, **kwargs)
