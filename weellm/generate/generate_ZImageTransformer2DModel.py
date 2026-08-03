import torch
import numpy as np
from PIL import Image
from typing import Optional, Union
from weellm.utils import report_memory, clean_memory

def generate(
    self,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None,
    **kwargs,
) -> Image.Image:
    print(f"\nGenerating {width}x{height} -- {num_inference_steps} steps ...")

    device = self.device
    dtype  = self.dtype

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    print("  [1/3] Encoding prompt ...")
    report_memory("Before text encode")

    tokenizer = self._tokenizer
    messages = [{"role": "user", "content": prompt}]
    prompt_with_template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    text_inputs = tokenizer(
        [prompt_with_template],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids    = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    raw_embeds = self._text_encoder.encode_ids(input_ids, attention_mask=prompt_masks)
    cap_feats = [raw_embeds[0][prompt_masks[0]]]

    report_memory("After text encode")

    vae_scale = self._vae_scale_factor * 2
    lat_h = 2 * (height // vae_scale)
    lat_w = 2 * (width  // vae_scale)
    in_channels = self._transformer_streamer.model.config.in_channels

    latents = torch.randn(
        (1, in_channels, lat_h, lat_w),
        device=device, dtype=torch.float32, generator=generator,
    )

    image_seq_len = (lat_h // 2) * (lat_w // 2)

    def _calculate_shift(image_seq_len, base=256, max_sl=4096,
                          base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_sl - base)
        b = base_shift - m * base
        return image_seq_len * m + b

    def _get_default_sigmas(n):
        return torch.linspace(1.0, 1.0 / n, n).tolist()

    mu = _calculate_shift(
        image_seq_len,
        self._scheduler.config.get("base_image_seq_len", 256),
        self._scheduler.config.get("max_image_seq_len", 4096),
        self._scheduler.config.get("base_shift", 0.5),
        self._scheduler.config.get("max_shift", 1.15),
    )
    sigmas = _get_default_sigmas(num_inference_steps)
    try:
        self._scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    except Exception:
        self._scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self._scheduler.timesteps
    if hasattr(self._scheduler, "set_begin_index"):
        self._scheduler.set_begin_index(0)

    t_scale = getattr(self._transformer_streamer.model.config, "t_scale", 1000.0)

    print("  [2/3] Denoising ...")
    for i, t in enumerate(timesteps):
        print(f"        Step {i+1}/{num_inference_steps} (t={t:.4f}) ...")
        report_memory(f"  step {i+1}")

        lat_bf = latents.to(dtype)
        x_list = [lat_bf[b].unsqueeze(1) for b in range(lat_bf.shape[0])]

        t_batch = t.expand(latents.shape[0])
        t_norm  = ((1000.0 - t_batch) / t_scale).to(dtype=dtype, device=device)

        output = self._transformer_streamer.model(
            x=x_list,
            t=t_norm,
            cap_feats=cap_feats,
            return_dict=False,
        )
        noise_pred = torch.stack(
            [out.squeeze(1) for out in output[0]], dim=0
        ).float()

        noise_pred = -noise_pred

        latents = self._scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]

    print("  [3/3] Decoding ...")
    scaling   = self._vae.config.scaling_factor
    shift     = getattr(self._vae.config, "shift_factor", 0.0)
    latents_for_vae = (latents.to(dtype) / scaling) + shift
    image = self._vae.decode(latents_for_vae, return_dict=False)[0]
    report_memory("After VAE decode")

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype(np.uint8)
    image = Image.fromarray(image[0])

    print("  Done!\n")
    return image

