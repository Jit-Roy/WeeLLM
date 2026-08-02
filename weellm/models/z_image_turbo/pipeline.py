"""
pipeline.py -- Memory-frugal Z-Image-Turbo pipeline using hook-based streaming.

Architecture:
  - ZImageTransformer2DModel (30 main layers + 2 context + 2 noise refiner)
    cap_feat_dim = 2560 (expects single-layer Qwen3 hidden states)
  - Qwen3ForCausalLM text encoder (36 layers, hidden_size=2560)
    We extract ONLY the last layer (layer 35) → (B, seq, 2560)
  - AutoencoderKL VAE
  - FlowMatchEulerDiscreteScheduler (shift=3.0)

VRAM budget:
  - VAE:           ~160 MB (resident)
  - TE resident:   ~45 MB  (resident)
  - Transformer resident: ~200 MB (resident)
  - One streamed transformer layer: ~800 MB
  - Peak: ~1.5-2.5 GB (within 4 GB budget)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .transformer_streamer import ZImageStreamer
# Reuse Flux2 Klein's Qwen3 text encoder streamer -- same architecture
from weellm.models.flux2_klein.text_encoder_streamer import StreamingQwen3TextEncoder
from weellm.core.base_pipeline import BasePipeline
from weellm.core.utils import clean_memory, report_memory


# The original ZImagePipeline uses hidden_states[-2] (second-to-last = layer 34 of 36)
# and masks out padding tokens to give variable-length embeddings per item.
_ZIMAGE_EXTRACT_LAYERS = (34,)


class WeeZImageTurboPipeline(BasePipeline):
    """
    Memory-frugal Z-Image-Turbo pipeline.

    Streams the ZImageTransformer2DModel and Qwen3 text encoder
    one layer at a time from disk, staying within 4 GB VRAM.
    """

    def __init__(
        self,
        scheduler,
        vae: nn.Module,
        text_encoder: StreamingQwen3TextEncoder,
        transformer: ZImageStreamer,
        vae_scale_factor: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.scheduler      = scheduler
        self.vae            = vae
        self.text_encoder   = text_encoder
        self.transformer    = transformer
        self.vae_sf         = vae_scale_factor
        self.device         = device
        self.dtype          = dtype

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs,
    ) -> "WeeZImageTurboPipeline":
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

        model_dir = Path(model_dir)

        print("\n" + "=" * 60)
        print("  WeeZImageTurboPipeline -- initialising")
        print("=" * 60 + "\n")

        # [1] Scheduler
        print("[1/4] Loading scheduler ...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_dir / "scheduler")
        )
        print("      OK\n")

        # [2] VAE
        print("[2/4] Loading VAE (~160 MB, resident on GPU) ...")
        vae = AutoencoderKL.from_pretrained(
            str(model_dir / "vae"), torch_dtype=dtype
        ).to(device)
        vae.eval()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        report_memory("After VAE load")
        print()

        # [3] Streaming text encoder (Qwen3, 36 layers, hidden=2560)
        # We extract only the LAST layer to match cap_feat_dim=2560.
        print("[3/4] Preparing streaming Qwen3 text encoder ...")
        text_encoder = StreamingQwen3TextEncoder(
            text_encoder_dir=model_dir / "text_encoder",
            tokenizer_dir=model_dir / "tokenizer",
            device=device,
            dtype=dtype,
            extract_layers=_ZIMAGE_EXTRACT_LAYERS,  # only layer 35 → (B, seq, 2560)
            max_length=512,
        )
        text_encoder._ensure_initialized()
        report_memory("After text encoder init")
        print()

        # [4] Streaming transformer
        print("[4/4] Preparing streaming ZImage transformer ...")
        transformer = ZImageStreamer.from_pretrained(
            transformer_dir=model_dir / "transformer",
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )
        report_memory("After transformer init")

        print("\n" + "=" * 60)
        print("  Pipeline ready.")
        print("=" * 60 + "\n")

        return cls(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            vae_scale_factor=vae_scale_factor,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
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

        # ── Seeding ─────────────────────────────────────────────────────
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # ── [1] Text encoding ───────────────────────────────────────────
        # The original ZImagePipeline uses:
        #   1. Qwen3 chat template with enable_thinking=True
        #   2. hidden_states[-2]  (second-to-last layer, index 34 of 36)
        #   3. Masks out padding tokens → variable-length per-item tensor
        print("  [1/3] Encoding prompt ...")
        report_memory("Before text encode")

        # Apply Qwen3 chat template (matches original pipeline exactly)
        tokenizer = self.text_encoder.tokenizer
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

        # Run the streaming text encoder — extract layer 34 (hidden_states[-2])
        # Our StreamingQwen3TextEncoder stores hidden states for specified layers.
        # Set extract_layers=(34,) so it returns (1, seq, 2560) from layer 34.
        raw_embeds = self.text_encoder.encode_ids(input_ids, attention_mask=prompt_masks)
        # raw_embeds shape: (1, seq_len, 2560)

        # Mask out padding → variable-length tensor (matches original pipeline)
        # cap_feats is a list with one (non_pad_len, 2560) tensor per batch item
        cap_feats = [raw_embeds[0][prompt_masks[0]]]   # [(valid_tokens, 2560)]

        report_memory("After text encode")

        # ── [2] Prepare latents ─────────────────────────────────────────
        # Shape follows original: 2 * (H // (vae_sf * 2)), same for W
        vae_scale = self.vae_sf * 2   # = 16 for standard VAE
        lat_h = 2 * (height // vae_scale)   # 512 → 64
        lat_w = 2 * (width  // vae_scale)   # 512 → 64
        in_channels = self.transformer.model.config.in_channels   # 16

        # Original uses float32 latents, keeps them in float32 throughout scheduler
        latents = torch.randn(
            (1, in_channels, lat_h, lat_w),
            device=device, dtype=torch.float32, generator=generator,
        )

        # ── [3] Scheduler timesteps ─────────────────────────────────────
        # Use original pipeline's custom sigmas + mu for proper shift
        image_seq_len = (lat_h // 2) * (lat_w // 2)   # tokens after patch_size=2

        def _calculate_shift(image_seq_len, base=256, max_sl=4096,
                              base_shift=0.5, max_shift=1.15):
            m = (max_shift - base_shift) / (max_sl - base)
            b = base_shift - m * base
            return image_seq_len * m + b

        def _get_default_sigmas(n):
            return torch.linspace(1.0, 1.0 / n, n).tolist()

        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = _get_default_sigmas(num_inference_steps)
        try:
            self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        except Exception:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        t_scale = getattr(self.transformer.model.config, "t_scale", 1000.0)

        # ── [4] Denoising loop ──────────────────────────────────────────
        print("  [2/3] Denoising ...")
        for i, t in enumerate(timesteps):
            print(f"        Step {i+1}/{num_inference_steps} (t={t:.4f}) ...")
            report_memory(f"  step {i+1}")

            # Convert (B, C, H, W) → list of (C, F=1, H, W)
            lat_bf = latents.to(dtype)
            x_list = [lat_bf[b].unsqueeze(1) for b in range(lat_bf.shape[0])]

            # CRITICAL: timestep normalization from original pipeline:
            #   timestep = (1000 - scheduler_t) / 1000
            #   At t=1000 (pure noise): 0.0  → model knows: "all noise"
            #   At t=0    (clean):      1.0  → model knows: "fully clean"
            t_batch = t.expand(latents.shape[0])
            t_norm  = ((1000.0 - t_batch) / t_scale).to(dtype=dtype, device=device)

            output = self.transformer.model(
                x=x_list,
                t=t_norm,
                cap_feats=cap_feats,
                return_dict=False,
            )
            # output[0] = list of (C, F, H, W) tensors
            # Stack → (B, C, F, H, W) → squeeze F → (B, C, H, W)
            noise_pred = torch.stack(
                [out.squeeze(1) for out in output[0]], dim=0
            ).float()   # (B, C, H, W), float32

            # CRITICAL: original pipeline negates the model output!
            noise_pred = -noise_pred

            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]   # latents stay float32

        # ── [5] VAE decode ──────────────────────────────────────────────
        print("  [3/3] Decoding ...")
        # Original: latents = (latents / scaling_factor) + shift_factor
        scaling   = self.vae.config.scaling_factor    # 0.3611
        shift     = getattr(self.vae.config, "shift_factor", 0.0)  # 0.1159
        latents_for_vae = (latents.to(dtype) / scaling) + shift
        image = self.vae.decode(latents_for_vae, return_dict=False)[0]
        report_memory("After VAE decode")

        # ── [6] Post-process ─────────────────────────────────────────────
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype(np.uint8)
        image = Image.fromarray(image[0])

        print("  Done!\n")
        return image

    @property
    def model_name(self) -> str:
        return "WeeZImageTurboPipeline"
