"""
generate_HiDreamImageTransformer2DModel.py
Provides generation logic for HiDreamImagePipeline within the WeeLLM system.
"""

import math
import torch
import torch.nn as nn
from tqdm import tqdm

from weellm.utils import clean_memory
from diffusers.utils.torch_utils import randn_tensor

import diffusers.models.transformers.transformer_hidream_image as hidream_mod

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

@torch.no_grad()
def generate(
    pipeline,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    generator: torch.Generator = None,
    **kwargs
):
    device = pipeline.device
    dtype = pipeline.dtype
    print(f"\n[DEBUG] Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.3f} GB")
    
    # 1) VAE Offload (User's brilliant suggestion!)
    print(f"Offloading VAE to CPU to save ~1.26 GB of VRAM...")
    pipeline._vae.to("cpu")
    clean_memory(device)
    print(f"[DEBUG] After VAE Offload. VRAM: {torch.cuda.memory_allocated()/1e9:.3f} GB")
    
    # Text encoders
    te_clip_1 = pipeline._text_encoder_1
    tok_clip_1 = pipeline._tokenizer_1
    
    te_clip_2 = pipeline._text_encoder_2
    tok_clip_2 = pipeline._tokenizer_2
    
    te_t5 = pipeline._text_encoder_3
    tok_t5 = pipeline._tokenizer_3
    
    te_llama = pipeline._text_encoder_4
    tok_llama = pipeline._tokenizer_4
    
    do_classifier_free_guidance = guidance_scale > 1.0

    print("Encoding prompt using CLIP 1...")
    text_inputs = tok_clip_1(
        prompt, padding="max_length", max_length=min(128, 218), truncation=True, return_tensors="pt"
    )
    prompt_embeds_1 = te_clip_1(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled_1 = prompt_embeds_1[0].to(dtype=dtype)
    if do_classifier_free_guidance:
        uncond_inputs = tok_clip_1(
            "", padding="max_length", max_length=min(128, 218), truncation=True, return_tensors="pt"
        )
        uncond_embeds_1 = te_clip_1(uncond_inputs.input_ids.to(device), output_hidden_states=True)
        uncond_pooled_1 = uncond_embeds_1[0].to(dtype=dtype)
    clean_memory(device)

    print("Encoding prompt using CLIP 2...")
    text_inputs_2 = tok_clip_2(
        prompt, padding="max_length", max_length=min(128, 218), truncation=True, return_tensors="pt"
    )
    prompt_embeds_2 = te_clip_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
    pooled_2 = prompt_embeds_2[0].to(dtype=dtype)
    if do_classifier_free_guidance:
        uncond_inputs_2 = tok_clip_2(
            "", padding="max_length", max_length=min(128, 218), truncation=True, return_tensors="pt"
        )
        uncond_embeds_2 = te_clip_2(uncond_inputs_2.input_ids.to(device), output_hidden_states=True)
        uncond_pooled_2 = uncond_embeds_2[0].to(dtype=dtype)
    clean_memory(device)

    print("Encoding prompt using T5...")
    text_inputs_3 = tok_t5(
        prompt, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt"
    )
    prompt_embeds_t5 = te_t5(text_inputs_3.input_ids.to(device), attention_mask=text_inputs_3.attention_mask.to(device))[0].to(dtype=dtype)
    if do_classifier_free_guidance:
        uncond_inputs_3 = tok_t5(
            "", padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt"
        )
        negative_prompt_embeds_t5 = te_t5(uncond_inputs_3.input_ids.to(device), attention_mask=uncond_inputs_3.attention_mask.to(device))[0].to(dtype=dtype)
    clean_memory(device)

    print("Encoding prompt using LLaMA...")
    if tok_llama.pad_token is None:
        tok_llama.pad_token = tok_llama.eos_token
    text_inputs_4 = tok_llama(
        prompt, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt"
    )
    # LLaMA streamer's encode_ids returns stacked hidden states directly
    prompt_embeds_llama3 = te_llama.encode_ids(text_inputs_4.input_ids, attention_mask=text_inputs_4.attention_mask)
    if do_classifier_free_guidance:
        uncond_inputs_4 = tok_llama(
            "", padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt"
        )
        negative_prompt_embeds_llama3 = te_llama.encode_ids(uncond_inputs_4.input_ids, attention_mask=uncond_inputs_4.attention_mask)
    clean_memory(device)
    pipeline.free_text_encoder_ram()
    
    print("Offloading Text Encoders from VRAM to CPU to save 1.7+ GB ...")
    from accelerate.utils.modeling import set_module_tensor_to_device
    for te in [te_clip_1, te_clip_2, te_t5, te_llama]:
        if te is not None:
            model = getattr(te, "model", getattr(te, "_model", te))
            if model is not None:
                for name, param in list(model.named_parameters()) + list(model.named_buffers()):
                    if param is not None and param.device.type != "meta":
                        set_module_tensor_to_device(model, name, "cpu")
    clean_memory(device)

    # Combine pooled embeds
    pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = torch.cat([uncond_pooled_1, uncond_pooled_2], dim=-1)
        
        prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5], dim=0)
        prompt_embeds_llama3 = torch.cat([negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    print("Initializing Latents...")
    vae_scale_factor = pipeline._vae_scale_factor
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    
    division = vae_scale_factor * 2
    default_sample_size = 128
    S_max = (default_sample_size * vae_scale_factor) ** 2
    scale = S_max / (width * height)
    scale = math.sqrt(scale)
    width, height = int(width * scale // division * division), int(height * scale // division * division)

    num_channels_latents = pipeline._transformer.config.in_channels
    shape = (1, num_channels_latents, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
    print("Preparing Scheduler...")
    # set timesteps
    image_seq_len = (height // (pipeline.vae.config.block_out_channels[-1] if hasattr(pipeline.vae, "config") else 8) // 2) * \
                    (width // (pipeline.vae.config.block_out_channels[-1] if hasattr(pipeline.vae, "config") else 8) // 2)
    mu = calculate_shift(image_seq_len)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = pipeline.scheduler.timesteps
    
    print("Denoising...")
    print(f"\n[DEBUG] Starting Denoising Loop. Latents prepared. VRAM: {torch.cuda.memory_allocated()/1e9:.3f} GB")
    import time
    for i, t in enumerate(tqdm(timesteps)):
        step_start = time.time()
        print(f"\n[DEBUG] --- Step {i+1}/{num_inference_steps} ---")
        print(f"[DEBUG] Pre-forward VRAM: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        timestep = t.expand(latent_model_input.shape[0])
        
        print(f"[DEBUG] Starting Transformer forward pass (Streaming active)...")
        # get noise pred
        noise_pred = pipeline._transformer(
            hidden_states=latent_model_input,
            timesteps=timestep,
            encoder_hidden_states_t5=prompt_embeds_t5,
            encoder_hidden_states_llama3=prompt_embeds_llama3,
            pooled_embeds=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        
        print(f"[DEBUG] Post-forward VRAM (Before scheduler step): {torch.cuda.memory_allocated()/1e9:.3f} GB")
        noise_pred = -noise_pred

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"[DEBUG] End of Step {i+1} VRAM: {torch.cuda.memory_allocated()/1e9:.3f} GB (Took {time.time()-step_start:.2f}s)")

    # Free memory before VAE decode
    import gc
    del noise_pred
    del prompt_embeds_llama3
    del prompt_embeds_t5
    del pooled_prompt_embeds
    clean_memory(device)
    
    print("Decoding Image...")
    # Move VAE back to GPU for decoding
    print("Loading VAE back to GPU...")
    pipeline._vae.to(device)
    
    # use tiling to save VRAM
    pipeline._vae.enable_tiling()
    
    # HiDream unscaling
    scaling_factor = pipeline._vae.config.scaling_factor
    shift_factor = getattr(pipeline._vae.config, "shift_factor", None) or 0.0
    latents = (latents / scaling_factor) + shift_factor
    
    image = pipeline._vae.decode(latents, return_dict=False)[0]
    
    # Process image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    
    import numpy as np
    from PIL import Image
    image = (image * 255).round().astype(np.uint8)
    image = [Image.fromarray(img) for img in image]
    
    return image[0]
