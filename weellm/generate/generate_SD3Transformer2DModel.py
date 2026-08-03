import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

def _sd3_encode_prompt(self, prompt: str, max_t5_length: int=256):
    """
        Returns:
          prompt_embeds:        (1, seq_len, 4096) from T5
          pooled_prompt_embeds: (1, 2048)           from concat(CLIP-1, CLIP-2)
        """
    clip1_inputs = self._tokenizer(prompt, padding='max_length', max_length=self._tokenizer.model_max_length, truncation=True, return_tensors='pt').to(self.device)
    with torch.no_grad():
        clip1_out = self._text_encoder(clip1_inputs.input_ids, output_hidden_states=True)
    clip1_hidden = clip1_out.hidden_states[-2]
    clip1_pooled = clip1_out.text_embeds
    del clip1_inputs, clip1_out
    clip2_inputs = self._tokenizer_2(prompt, padding='max_length', max_length=self._tokenizer_2.model_max_length, truncation=True, return_tensors='pt').to(self.device)
    with torch.no_grad():
        clip2_out = self._text_encoder_2(clip2_inputs.input_ids, output_hidden_states=True)
    clip2_hidden = clip2_out.hidden_states[-2]
    clip2_pooled = clip2_out.text_embeds
    del clip2_inputs, clip2_out
    clip_combined = torch.cat([clip1_hidden, clip2_hidden], dim=-1)
    pooled_prompt_embeds = torch.cat([clip1_pooled, clip2_pooled], dim=-1).to(dtype=self.dtype)
    del clip1_hidden, clip2_hidden, clip1_pooled, clip2_pooled
    t5_inputs = self._tokenizer_3(prompt, padding='max_length', max_length=max_t5_length, truncation=True, return_tensors='pt').to(self.device)
    t5_out = self._text_encoder_3(t5_inputs.input_ids)
    t5_hidden = t5_out[0].to(dtype=self.dtype)
    del t5_inputs, t5_out
    clip_padded = torch.nn.functional.pad(clip_combined, (0, 4096 - clip_combined.shape[-1])).to(dtype=self.dtype)
    prompt_embeds = torch.cat([clip_padded, t5_hidden], dim=1)
    del clip_combined, clip_padded, t5_hidden
    clean_memory(self.device)
    return (prompt_embeds, pooled_prompt_embeds)

def generate(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=28, guidance_scale: float=4.5, seed: int=42, max_t5_length: int=256) -> Image.Image:
    generator = torch.Generator(device=self.device).manual_seed(seed)
    print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps (guidance={guidance_scale}) ...')
    print('  [1/3] Encoding prompt ...')
    report_memory('Before text encode')
    prompt_embeds, pooled_prompt_embeds = self._sd3_encode_prompt(prompt, max_t5_length)
    negative_prompt_embeds, negative_pooled = self._sd3_encode_prompt('', max_t5_length)
    report_memory('After text encode')
    latent_h = height // self._vae_scale_factor
    latent_w = width // self._vae_scale_factor
    latents = torch.randn((1, 16, latent_h, latent_w), generator=generator, device=self.device, dtype=self.dtype)
    self._scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps = self._scheduler.timesteps
    print('  [2/3] Denoising ...')
    for i, t in enumerate(timesteps):
        print(f'        Step {i + 1}/{num_inference_steps} (t={t.item():.1f}) ...')
        latent_model_input = torch.cat([latents, latents])
        timestep = t.expand(2).long()
        enc_hidden = torch.cat([negative_prompt_embeds, prompt_embeds])
        enc_pooled = torch.cat([negative_pooled, pooled_prompt_embeds])
        noise_pred = self._transformer(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=enc_hidden, pooled_projections=enc_pooled, return_dict=False)[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        report_memory(f'  step {i + 1}')
    print('  [3/3] Decoding ...')
    latents = latents / self._vae.config.scaling_factor + self._vae.config.shift_factor
    image = self._vae.decode(latents, return_dict=False)[0]
    report_memory('After VAE decode')
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    print('  Done!')
    return Image.fromarray(image)

