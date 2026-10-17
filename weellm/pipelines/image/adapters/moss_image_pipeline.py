"""
moss_image_pipeline.py -- MOSSImagePipeline and MOSSImageScheduler adapter for WeeLLM.

Provides:
  - MOSSImageScheduler: Euler integration using training project's V2 nearest-grid schedule
  - MOSSImagePipeline: Diffusers-compatible text-to-image pipeline for MOSS-Image models
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from diffusers import AutoencoderKLFlux2, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer

from weellm.models.transformers.moss_image_transformer_2d_model import (
    MOSSImageTransformer,
    _validate_skip_layers,
)
from weellm.schedulers.moss_image_scheduler import MOSSImageScheduler


class MOSSImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae"]

    def __init__(
        self,
        transformer: MOSSImageTransformer,
        scheduler: MOSSImageScheduler,
        text_encoder=None,
        tokenizer=None,
        vae=None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length: int = 4096):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text encoder/tokenizer required for string prompts.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        rows = []
        for value in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_sequence_length,
            ).to(self._execution_device)
            output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, :row.shape[0]] = row
            mask[index, :row.shape[0]] = True
        return embeds, mask

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        max_sequence_length: int = 4096,
        output_type: str = "pil",
        return_dict: bool = True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale: float = 0.0,
        slg_start: float = 0.0,
        slg_stop: float = 1.0,
        **kwargs,
    ):
        uncond_skip_layers = _validate_skip_layers(
            uncond_skip_layers, len(self.transformer.layers), "uncond_skip_layers"
        )
        slg_layers = _validate_skip_layers(slg_layers, len(self.transformer.layers), "slg_layers")
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative.")
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1.")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be pil, np, pt or latent.")
        if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Invalid image count or guidance scale.")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096.")
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds.")

        device = self._execution_device
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
        batch = prompt_embeds.shape[0]

        def prepare(embeds, mask):
            if embeds.ndim != 3 or embeds.shape[0] != batch:
                raise ValueError("Prompt embeddings must be [B, S, C] with matching batch size.")
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
            return (
                embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
                mask.to(device).repeat_interleave(num_images_per_prompt, 0),
            )

        positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
        do_cfg = guidance_scale != 1.0
        if do_cfg:
            if negative_prompt_embeds is None:
                negatives = [negative_prompt] * batch if isinstance(negative_prompt, str) else negative_prompt
                if len(negatives) != batch:
                    raise ValueError("negative_prompt must match prompt batch size.")
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    negatives, max_sequence_length
                )
            negative, negative_mask = prepare(negative_prompt_embeds, negative_prompt_attention_mask)

        shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"latents must have shape {shape}.")
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_steps = len(self.scheduler.timesteps)

        for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            skip_active = slg_start <= step_index / num_steps < slg_stop
            prediction = self.transformer(latents, timestep, positive, positive_mask).sample.float()
            conditional = prediction
            if do_cfg:
                skip_kwargs = {"skip_layers": uncond_skip_layers} if uncond_skip_layers and skip_active else {}
                uncond = self.transformer(latents, timestep, negative, negative_mask, **skip_kwargs).sample.float()
                prediction = uncond + guidance_scale * (prediction - uncond)
            if slg_layers and slg_scale > 0 and skip_active:
                skipped = self.transformer(
                    latents, timestep, positive, positive_mask, skip_layers=slg_layers
                ).sample.float()
                if not do_cfg:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            latents = self.scheduler.step(prediction, timestep, latents, return_dict=False)[0]

        if output_type == "latent":
            images = latents
        else:
            if self.vae is None:
                raise ValueError("A FLUX.2 VAE is required for decoded output.")
            z = latents.to(dtype=self.vae.dtype)
            bn = self.vae.bn
            eps = self.vae.config.batch_norm_eps
            z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
            images = self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
            images = self.image_processor.postprocess(images, output_type=output_type)

        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=images) if return_dict else (images,)
