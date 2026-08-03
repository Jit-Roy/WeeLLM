import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

def _qwen_pack_latents(self, latents: torch.Tensor, patch_size: int=2) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W/p^2, C*p^2)"""
    B, C, H, W = latents.shape
    ph = H // patch_size
    pw = W // patch_size
    latents = latents.view(B, C, ph, patch_size, pw, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, ph * pw, C * patch_size * patch_size)
    return latents

def _qwen_unpack_latents(self, latents: torch.Tensor, height: int, width: int, patch_size: int=2) -> torch.Tensor:
    """(B, seq, C*p^2) -> (B, C, H, W)"""
    B = latents.shape[0]
    ph = height // self._vae_scale_factor // patch_size
    pw = width // self._vae_scale_factor // patch_size
    C = latents.shape[-1] // (patch_size * patch_size)
    latents = latents.view(B, ph, pw, C, patch_size, patch_size)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, C, ph * patch_size, pw * patch_size)
    return latents

def generate(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=50, true_cfg_scale: float=4.0, negative_prompt: str='', seed: int=42, max_sequence_length: int=512) -> Image.Image:
    generator = torch.Generator(device=self.device).manual_seed(seed)
    do_cfg = true_cfg_scale > 1.0
    print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps (cfg={true_cfg_scale}) ...')
    print('  [1/3] Encoding prompt ...')
    report_memory('Before text encode')
    prompt_embeds, prompt_embeds_mask = self._text_encoder.encode(prompt)
    if do_cfg:
        neg_embeds, neg_mask = self._text_encoder.encode(negative_prompt)
    report_memory('After text encode')
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
            return (emb, mask)
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        if neg_mask is None:
            neg_mask = torch.ones(neg_embeds.shape[:2], dtype=neg_embeds.dtype, device=neg_embeds.device)
        prompt_embeds, prompt_embeds_mask = pad_seq(prompt_embeds, prompt_embeds_mask, max_len)
        neg_embeds, neg_mask = pad_seq(neg_embeds, neg_mask, max_len)
        prompt_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
        prompt_embeds_mask = torch.cat([prompt_embeds_mask, neg_mask], dim=0)
    num_channels_latents = self._transformer.model.config.in_channels // 4
    latent_h = height // self._vae_scale_factor
    latent_w = width // self._vae_scale_factor
    latents = torch.randn((1, num_channels_latents, latent_h, latent_w), generator=generator, device=self.device, dtype=self.dtype)
    latents = self._qwen_pack_latents(latents, patch_size=2)
    img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
    if do_cfg:
        img_shapes = img_shapes * 2
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(image_seq_len, base_seq_len=self._scheduler.config.get('base_image_seq_len', 256), max_seq_len=self._scheduler.config.get('max_image_seq_len', 4096), base_shift=self._scheduler.config.get('base_shift', 0.5), max_shift=self._scheduler.config.get('max_shift', 1.15))
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    self._scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
    timesteps = self._scheduler.timesteps
    print('  [2/3] Denoising ...')
    for i, t in enumerate(timesteps):
        print(f'        Step {i + 1}/{num_inference_steps} (t={t.item():.1f}) ...')
        latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        timestep = t.expand(latent_model_input.shape[0]).to(self.dtype)
        with self._transformer.cache_context('forward'):
            noise_pred_all = self._transformer(hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None, encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_embeds_mask, img_shapes=img_shapes, return_dict=False)[0]
        if do_cfg:
            noise_pred, neg_noise_pred = noise_pred_all.chunk(2)
            comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb, dim=-1, keepdim=True)
            noise_pred = comb * (cond_norm / comb_norm)
        else:
            noise_pred = noise_pred_all
        latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        report_memory(f'  step {i + 1}')
    print('  [3/3] Decoding ...')
    latents = self._qwen_unpack_latents(latents, height, width, patch_size=2)
    latents = latents.to(self._vae.dtype)
    latents_mean = torch.tensor(self._vae.config.latents_mean).view(1, self._vae.config.z_dim, 1, 1).to(device=self.device, dtype=latents.dtype)
    latents_std_inv = 1.0 / torch.tensor(self._vae.config.latents_std).view(1, self._vae.config.z_dim, 1, 1).to(device=self.device, dtype=latents.dtype)
    latents = latents / latents_std_inv + latents_mean
    latents = latents.unsqueeze(2)
    self._vae.enable_tiling()
    image = self._vae.decode(latents, return_dict=False)[0][:, :, 0]
    self._vae.disable_tiling()
    report_memory('After VAE decode')
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    print('  Done!')
    return Image.fromarray(image)

