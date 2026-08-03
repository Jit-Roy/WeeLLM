import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

def _flux2_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)"""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)

def _flux2_prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> latent_ids (B, H*W, 4) with [t, h, w, l] coords."""
    B, _, H, W = latents.shape
    t = torch.arange(1)
    h = torch.arange(H)
    w = torch.arange(W)
    l = torch.arange(1)
    ids = torch.cartesian_prod(t, h, w, l)
    return ids.unsqueeze(0).expand(B, -1, -1)

def _flux2_prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """(B, L, D) -> text_ids (B, L, 4) with [t=0, h=0, w=0, l=i] coords."""
    B, L, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)

def _flux2_unpack_latents(x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scatter sequence tokens back into (B, C, H, W) using position IDs."""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].long()
        w_ids = pos[:, 2].long()
        flat_ids = h_ids * width + w_ids
        out = torch.zeros((height * width, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(height, width, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)

def _flux2_unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    B, C4, H, W = latents.shape
    C = C4 // 4
    latents = latents.reshape(B, C, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(B, C, H * 2, W * 2)
    return latents

@torch.no_grad()
def generate(
    self,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    output_type: str = "pil",
) -> Union[Image.Image, torch.Tensor]:
    generator = (
        torch.Generator(device=self.device).manual_seed(seed)
        if seed is not None
        else None
    )

    print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")
    prompt_embeds = self._text_encoder.encode(prompt)
    clean_memory(self.device)
    report_memory("After text encode")

    text_ids = _flux2_prepare_text_ids(prompt_embeds).to(self.device)

    eff_h = 2 * (int(height) // (self._vae_scale_factor * 2))
    eff_w = 2 * (int(width)  // (self._vae_scale_factor * 2))
    num_latent_channels = 32

    raw_latent_shape = (1, num_latent_channels * 4, eff_h // 2, eff_w // 2)
    latents_raw = torch.randn(
        raw_latent_shape, dtype=self.dtype, device=self.device, generator=generator
    )
    latent_ids = _flux2_prepare_latent_ids(latents_raw).to(self.device)
    latents = _flux2_pack_latents(latents_raw)
    del latents_raw

    print("  [2/3] Denoising ...")
    image_seq_len = latents.shape[1]
    try:
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self._scheduler, num_inference_steps, self.device, sigmas=sigmas, mu=mu
        )
    except Exception:
        self._scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self._scheduler.timesteps

    for step_idx, t in enumerate(timesteps):
        print(f"        Step {step_idx + 1}/{num_inference_steps} (t={t.item():.4f}) ...")
        report_memory(f"  step {step_idx+1}")

        latent_model_input = latents.to(self.dtype)
        timestep = t.expand(1).to(self.dtype)

        noise_pred = self._transformer_streamer.model(
            hidden_states=latent_model_input,
            timestep=timestep / 1000.0,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            guidance=None,
            return_dict=False,
        )[0]

        latents = self._scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]
        clean_memory(self.device)

    if output_type == "tensor":
        return latents

    print("  [3/3] Decoding ...")
    latent_h = eff_h // 2
    latent_w = eff_w // 2

    latents_spatial = _flux2_unpack_latents(latents, latent_ids, latent_h, latent_w)

    bn_mean = self._vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents_spatial.device, dtype=latents_spatial.dtype
    )
    bn_std = torch.sqrt(
        self._vae.bn.running_var.view(1, -1, 1, 1)
        + self._vae.config.get("batch_norm_eps", 1e-4)
    ).to(device=latents_spatial.device, dtype=latents_spatial.dtype)
    latents_spatial = latents_spatial * bn_std + bn_mean

    latents_spatial = _flux2_unpatchify_latents(latents_spatial)

    latents_spatial = latents_spatial.to(dtype=self.dtype)
    image_tensor = self._vae.decode(latents_spatial, return_dict=False)[0]
    clean_memory(self.device)
    report_memory("After VAE decode")

    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = image_tensor[0].cpu().float().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).round().astype("uint8")
    image = Image.fromarray(image_np)

    print("  Done!\n")
    return image
