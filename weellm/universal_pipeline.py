import os
import json
import gc
import importlib
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from weellm.base_pipeline import BasePipeline
from weellm.utils import clean_memory, report_memory
import math

def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (math.log(max_seq_len) - math.log(base_seq_len))
    b = base_shift - m * math.log(base_seq_len)
    mu = m * math.log(image_seq_len) + b
    return mu


class UniversalWeePipeline(BasePipeline):
    def __init__(
        self,
        model_dir: Path,
        transformer,
        text_encoders: Dict[str, Any],
        tokenizers: Dict[str, Any],
        vae,
        scheduler,
        transformer_class_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        self.model_dir = model_dir
        self.transformer = transformer
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.vae = vae
        self.scheduler = scheduler
        self.transformer_class_name = transformer_class_name
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        **kwargs
    ):
        model_dir = Path(model_dir)
        index_path = model_dir / "model_index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
            
        print("\n============================================================")
        print(f"  UniversalWeePipeline -- Loading {index.get('_class_name', 'Unknown')}")
        print("============================================================\n")
        
        # 1. Tokenizers & Scheduler
        print("[1/4] Loading Tokenizers and Scheduler ...")
        tokenizers = {}
        for key in ["tokenizer", "tokenizer_2", "tokenizer_3"]:
            if key in index:
                from transformers import AutoTokenizer
                try:
                    tokenizers[key] = AutoTokenizer.from_pretrained(str(model_dir), subfolder=key)
                except Exception as e:
                    print(f"Warning: Failed to load {key}: {e}")

        scheduler_cls = getattr(importlib.import_module("diffusers"), index["scheduler"][1])
        scheduler = scheduler_cls.from_pretrained(str(model_dir), subfolder="scheduler")
        
        # 2. VAE (Resident on GPU)
        print("\n[2/4] Loading VAE (resident on GPU) ...")
        vae_cls = getattr(importlib.import_module("diffusers"), index["vae"][1])
        try:
            vae = vae_cls.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=dtype, variant="fp16", use_safetensors=True).to(device)
        except OSError:
            vae = vae_cls.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)

        # 3. Text Encoders (Streamed or Resident)
        print("\n[3/4] Preparing Text Encoders ...")
        TE_MAP = {
            "CLIPTextModel": "weellm.models.text_encoders.clip_text_model",
            "CLIPTextModelWithProjection": "weellm.models.text_encoders.clip_text_model",
            "T5EncoderModel": "weellm.models.text_encoders.t5_encoder_model",
            "Qwen2ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen3ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen2_5_VLForConditionalGeneration": "weellm.models.text_encoders.qwen2_5_vl_for_conditional_generation"
        }
        
        text_encoders = {}
        for key in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            if key in index:
                hf_cls_name = index[key][1]
                if hf_cls_name in TE_MAP:
                    module_path = TE_MAP[hf_cls_name]
                    streamer_cls_name = "CLIPTextModelStreamer" if "CLIP" in hf_cls_name else hf_cls_name + "Streamer"
                    module = importlib.import_module(module_path)
                    te_cls = getattr(module, streamer_cls_name)
                    
                    tok_key = key.replace("text_encoder", "tokenizer")
                    if "Qwen" in hf_cls_name:
                        text_encoders[key] = te_cls.from_pretrained(model_dir=str(model_dir / key), tokenizer=tokenizers.get(tok_key), device=device, dtype=dtype)
                    elif "CLIP" in hf_cls_name:
                        hf_module = importlib.import_module("transformers")
                        hf_cls = getattr(hf_module, hf_cls_name)
                        text_encoders[key] = te_cls.from_pretrained(hf_cls, str(model_dir), key, device=device, dtype=dtype, output_hidden_states=True)
                    else:
                        text_encoders[key] = te_cls.from_pretrained(model_dir=str(model_dir / key), device=device, dtype=dtype)
                else:
                    hf_module = importlib.import_module("transformers")
                    hf_cls = getattr(hf_module, hf_cls_name)
                    text_encoders[key] = hf_cls.from_pretrained(str(model_dir / key), torch_dtype=dtype).to(device)
                    text_encoders[key].eval()

        # 4. Transformer / UNet
        print("\n[4/4] Preparing Transformer / UNet ...")
        transformer_key = "transformer" if "transformer" in index else "unet"
        transformer_class_name = index[transformer_key][1]
        
        TR_MAP = {
            "FluxTransformer2DModel": "weellm.models.transformers.flux_transformer_2d_model",
            "Flux2Transformer2DModel": "weellm.models.transformers.flux2_transformer_2d_model",
            "ZImageTransformer2DModel": "weellm.models.transformers.z_image_transformer_2d_model",
            "SD3Transformer2DModel": "weellm.models.transformers.sd3_transformer_2d_model",
            "QwenImageTransformer2DModel": "weellm.models.transformers.qwen_image_transformer_2d_model",
            "UNet2DConditionModel": "weellm.models.unets.unet_2d_condition_model"
        }
        
        module_path = TR_MAP.get(transformer_class_name, "")
        if not module_path:
            raise ValueError(f"Unsupported architecture: {transformer_class_name}")
            
        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, transformer_class_name + "Streamer")
        
        if transformer_key == "unet":
            transformer = transformer_cls.from_pretrained(str(model_dir), device, dtype, prefetch)
        else:
            transformer = transformer_cls.from_pretrained(model_dir / transformer_key, device=device, dtype=dtype, prefetch=prefetch)
        
        print("\n============================================================")
        print("  UniversalWeePipeline ready.")
        print("============================================================\n")
        
        return cls(
            model_dir=model_dir,
            transformer=transformer,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            vae=vae,
            scheduler=scheduler,
            transformer_class_name=transformer_class_name,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
    @property
    def _text_encoder(self): return self.text_encoders.get("text_encoder")
    @property
    def _text_encoder_1(self): return self.text_encoders.get("text_encoder")
    @property
    def _text_encoder_2(self): return self.text_encoders.get("text_encoder_2")
    @property
    def _text_encoder_3(self): return self.text_encoders.get("text_encoder_3")
    @property
    def _tokenizer(self): return self.tokenizers.get("tokenizer")
    @property
    def _tokenizer_1(self): return self.tokenizers.get("tokenizer")
    @property
    def _tokenizer_2(self): return self.tokenizers.get("tokenizer_2")
    @property
    def _tokenizer_3(self): return self.tokenizers.get("tokenizer_3")
    @property
    def _vae(self): return self.vae
    @property
    def _scheduler(self): return self.scheduler
    @property
    def _transformer_streamer(self): return self.transformer
    @property
    def _transformer(self): return self.transformer
    @property
    def _unet(self): return self.transformer
    @property
    def _vae_scale_factor(self):
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels"):
            return 2 ** (len(self.vae.config.block_out_channels) - 1)
        return 8

    def generate(self, prompt: str, **kwargs):
        if "Flux" in self.transformer_class_name:
            return self._generate_flux(prompt, **kwargs)
        elif "ZImage" in self.transformer_class_name:
            return self._generate_zimage(prompt, **kwargs)
        elif "Qwen" in self.transformer_class_name:
            return self._generate_qwen(prompt, **kwargs)
        elif "SD3" in self.transformer_class_name:
            return self._generate_sd3(prompt, **kwargs)
        elif "UNet" in self.transformer_class_name:
            if "tokenizer_2" in self.tokenizers:
                return self._generate_sdxl(prompt, **kwargs)
            else:
                return self._generate_sd(prompt, **kwargs)
        else:
            raise ValueError(f"Cannot route generate for {self.transformer_class_name}")


    @staticmethod
    def _flux_pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    @staticmethod
    def _flux_prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> latent_ids (B, H*W, 4) with [t, h, w, l] coords."""
        B, _, H, W = latents.shape
        t = torch.arange(1)
        h = torch.arange(H)
        w = torch.arange(W)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, h, w, l)
        return ids.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _flux_prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def _flux_unpack_latents(x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
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

    @torch.no_grad()
    def _generate_flux(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=4, guidance_scale: float=1.0, seed: Optional[int]=None, output_type: str='pil') -> Union[Image.Image, torch.Tensor]:
        """
            Generate an image from a text prompt.

            Parameters
            ----------
            prompt : str
                Text description of the desired image.
            height, width : int
                Output dimensions (multiples of 16 recommended).
            num_inference_steps : int
                Number of denoising steps (4 for this distilled model).
            guidance_scale : float
                CFG scale.  1.0 disables classifier-free guidance (saves 2x
                transformer passes per step).
            seed : int, optional
                RNG seed for reproducibility.
            output_type : str
                "pil" -> PIL.Image.Image, "tensor" -> raw latents (B, C, H, W).

            Returns
            -------
            image : PIL.Image.Image or torch.Tensor
            """
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps ...')
        print('  [1/3] Encoding prompt ...')
        report_memory('Before text encode')
        if self._text_encoder_2 is not None:
            prompt_embeds, pooled_prompt_embeds = self._flux1_encode_prompt(prompt, max_t5_length=256)
        else:
            prompt_embeds = self._text_encoder.encode(prompt)
            pooled_prompt_embeds = None
        clean_memory(self.device)
        report_memory('After text encode')
        text_ids = self._flux_prepare_text_ids(prompt_embeds).to(self.device)
        eff_h = 2 * (int(height) // (self._vae_scale_factor * 2))
        eff_w = 2 * (int(width) // (self._vae_scale_factor * 2))
        num_latent_channels = 32
        raw_latent_shape = (1, num_latent_channels * 4, eff_h // 2, eff_w // 2)
        latents_raw = torch.randn(raw_latent_shape, dtype=self.dtype, device=self.device, generator=generator)
        latent_ids = self._flux_prepare_latent_ids(latents_raw).to(self.device)
        latents = self._flux_pack_latents(latents_raw)
        del latents_raw
        print('  [2/3] Denoising ...')
        image_seq_len = latents.shape[1]
        try:
            from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
            mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(self._scheduler, num_inference_steps, self.device, sigmas=sigmas, mu=mu)
        except Exception:
            self._scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self._scheduler.timesteps
        for step_idx, t in enumerate(timesteps):
            print(f'        Step {step_idx + 1}/{num_inference_steps} (t={t.item():.4f}) ...')
            report_memory(f'  step {step_idx + 1}')
            latent_model_input = latents.to(self.dtype)
            timestep = t.expand(1).to(self.dtype)
            kwargs = {}
            if pooled_prompt_embeds is not None:
                kwargs["pooled_projections"] = pooled_prompt_embeds
                
            noise_pred = self._transformer_streamer.model(
                hidden_states=latent_model_input, 
                timestep=timestep / 1000.0, 
                encoder_hidden_states=prompt_embeds, 
                txt_ids=text_ids, 
                img_ids=latent_ids, 
                guidance=None, 
                return_dict=False,
                **kwargs
            )[0]
            latents = self._scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            clean_memory(self.device)
        if output_type == 'tensor':
            return latents
        print('  [3/3] Decoding ...')
        latent_h = eff_h // 2
        latent_w = eff_w // 2
        latents_spatial = self._flux_unpack_latents(latents, latent_ids, latent_h, latent_w)
        bn_mean = self._vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents_spatial.device, dtype=latents_spatial.dtype)
        bn_std = torch.sqrt(self._vae.bn.running_var.view(1, -1, 1, 1) + self._vae.config.get('batch_norm_eps', 0.0001)).to(device=latents_spatial.device, dtype=latents_spatial.dtype)
        latents_spatial = latents_spatial * bn_std + bn_mean
        latents_spatial = self._flux_unpatchify_latents(latents_spatial)
        latents_spatial = latents_spatial.to(dtype=self.dtype)
        image_tensor = self._vae.decode(latents_spatial, return_dict=False)[0]
        clean_memory(self.device)
        report_memory('After VAE decode')
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor[0].cpu().float().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().astype('uint8')
        image = Image.fromarray(image_np)
        print('  Done!\n')
        return image

    @torch.no_grad()
    def _generate_zimage(
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

    @staticmethod
    def _flux_unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
            Reverse the 2x2 spatial packing applied before the transformer.
            (B, C*4, H, W) -> (B, C, H*2, W*2)
            Matches Flux2KleinPipeline._unpatchify_latents exactly.
            """
        B, C4, H, W = latents.shape
        C = C4 // 4
        latents = latents.reshape(B, C, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C, H * 2, W * 2)
        return latents

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

    @torch.no_grad()
    def _generate_qwen(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=50, true_cfg_scale: float=4.0, negative_prompt: str='', seed: int=42, max_sequence_length: int=512) -> Image.Image:
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

    @torch.no_grad()
    def _generate_sdxl(self, prompt: str, height: int=1024, width: int=1024, num_inference_steps: int=20, guidance_scale: float=5.0, seed: int=42) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps ...')
        print('  [1/3] Encoding prompt ...')
        report_memory('Before text encode')
        prompt_embeds, pooled_prompt_embeds = self._sdxl_encode_prompt(prompt)
        report_memory('After text encode')
        if guidance_scale > 1.0:
            uncond_embeds, uncond_pooled = self._sdxl_encode_prompt('')
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

    @torch.no_grad()
    def _generate_sd3(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=28, guidance_scale: float=4.5, seed: int=42, max_t5_length: int=256) -> Image.Image:
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
    def _generate_sd(self, prompt: str, height: int=512, width: int=512, num_inference_steps: int=20, guidance_scale: float=7.5, seed: int=42) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f'\nGenerating {width}x{height} -- {num_inference_steps} steps ...')
        print('  [1/3] Encoding prompt ...')
        report_memory('Before text encode')
        prompt_embeds = self._sd_encode_prompt(prompt)
        if guidance_scale > 1.0:
            uncond_embeds = self._sd_encode_prompt('')
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