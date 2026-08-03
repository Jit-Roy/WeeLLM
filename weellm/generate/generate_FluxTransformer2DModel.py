import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

def _flux1_pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    B, C, H, W = latents.shape
    p = patch_size
    latents = latents.reshape(B, C, H // p, p, W // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // p) * (W // p), C * p * p)
    return latents

def _flux1_unpack_latents(latents: torch.Tensor, height: int, width: int, patch_size: int = 2) -> torch.Tensor:
    B, seq_len, Cpp = latents.shape
    p = patch_size
    C = Cpp // (p * p)
    h = height // p
    w = width // p
    latents = latents.reshape(B, h, w, C, p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, C, h * p, w * p)
    return latents

def _flux1_prepare_latent_image_ids(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    h = height // 2
    w = width // 2
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]
    return ids.reshape(1, h * w, 3).expand(1, -1, -1)

def _flux1_prepare_text_ids(seq_len: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(1, seq_len, 3, device=device, dtype=dtype)

@torch.no_grad()
def _flux1_encode_prompt(self, prompt: str, max_t5_length: int = 256):
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
    pooled_prompt_embeds = clip_out[1].to(dtype=self.dtype)
    del clip_inputs, clip_out

    t5_inputs = self._tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_t5_length,
        truncation=True,
        return_tensors="pt",
    ).to(self.device)

    t5_out = self._text_encoder_2(t5_inputs.input_ids)
    prompt_embeds = t5_out[0].to(dtype=self.dtype)
    del t5_inputs, t5_out
    clean_memory(self.device)

    return prompt_embeds, pooled_prompt_embeds

@torch.no_grad()
def generate(
    self,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int = 42,
    max_t5_length: int = 256,
) -> Image.Image:

    generator = torch.Generator(device=self.device).manual_seed(seed)

    print(f"\\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")
    prompt_embeds, pooled_prompt_embeds = _flux1_encode_prompt(self, prompt, max_t5_length)
    report_memory("After text encode")
    if hasattr(self, 'free_text_encoder_ram'):
        self.free_text_encoder_ram()

    latent_h = height // 8
    latent_w = width // 8
    latents = torch.randn(
        (1, 16, latent_h, latent_w),
        generator=generator, device=self.device, dtype=self.dtype
    )

    packed_latents = _flux1_pack_latents(latents, patch_size=2)
    seq_len = packed_latents.shape[1]

    latent_image_ids = _flux1_prepare_latent_image_ids(latent_h, latent_w, self.device, self.dtype)
    text_ids = _flux1_prepare_text_ids(prompt_embeds.shape[1], self.device, self.dtype)

    if getattr(self._scheduler.config, "use_dynamic_shifting", False):
        m = (1.15 - 0.5) / (4096 - 256)
        b = 0.5 - m * 256
        mu = seq_len * m + b
        self._scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
    else:
        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        
    timesteps = self._scheduler.timesteps
    guidance = None
    if getattr(self._transformer.model.config, "guidance_embeds", False):
        guidance = torch.full([1], guidance_scale, device=self.device, dtype=self.dtype)
        guidance = guidance.expand(packed_latents.shape[0])

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
            guidance=guidance,
            return_dict=False,
        )[0]

        packed_latents = self._scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
        report_memory(f"  step {i+1}")

    print("  [3/3] Decoding ...")
    latents = _flux1_unpack_latents(packed_latents, latent_h, latent_w, patch_size=2)

    latents = (latents / self._vae.config.scaling_factor) + self._vae.config.shift_factor
    latents = latents.to(dtype=self.dtype)
    image = self._vae.decode(latents, return_dict=False)[0]
    report_memory("After VAE decode")

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype(np.uint8)

    print("  Done!")
    return Image.fromarray(image)

