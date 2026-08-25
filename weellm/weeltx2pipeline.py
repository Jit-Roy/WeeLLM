import os
import sys
import time
import logging
import gc
import torch
from weellm.pipeline import WeeBasePipeline

log = logging.getLogger("weellm")

DEFAULT_CACHE_DIR = r"D:\Personal Projects\LightLLM\.weellm_cache"

class WeeLTX2Pipeline(WeeBasePipeline):
    """
    LTX-2.5 22B Video inference pipeline.
    Streams Gemma3 12B and LTX-2.5 22B DiT on limited VRAM.
    """
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        vae_tile_size: int = 256,
        **kwargs,
    ) -> "WeeLTX2Pipeline":
        
        wrapper = cls.__new__(cls)
        object.__setattr__(wrapper, "_pipeline", None)
        object.__setattr__(wrapper, "model_dir", str(model_dir))
        object.__setattr__(wrapper, "device", device)
        object.__setattr__(wrapper, "torch_dtype", torch_dtype)
        object.__setattr__(wrapper, "cache_to_ram", cache_to_ram)
        
        vram_budget_gb = kwargs.pop("vram_budget_gb", None)
        if vram_budget_gb is None:
            vram_budget_gb = kwargs.pop("vram_budget", None)
            
        ram_budget_gb = kwargs.pop("ram_budget_gb", None)
        if ram_budget_gb is None:
            ram_budget_gb = kwargs.pop("ram_budget", None)
            
        if vram_budget_gb is not None or ram_budget_gb is not None:
            from weellm.models.base_streamer import BaseTransformerStreamer
            if vram_budget_gb is not None:
                BaseTransformerStreamer._global_vram_budget_gb = vram_budget_gb
                log.info(f"Set global VRAM budget to {vram_budget_gb} GB")
            if ram_budget_gb is not None:
                BaseTransformerStreamer._global_ram_budget_gb = ram_budget_gb
                log.info(f"Set global RAM budget to {ram_budget_gb} GB")
        
        try:
            from weellm.models.loras.lora_loader import MiniMaxH3LoRALoader
            lora_loader = MiniMaxH3LoRALoader()
            object.__setattr__(wrapper, "lora_loader", lora_loader)
        except Exception:
            object.__setattr__(wrapper, "lora_loader", None)
            
        return wrapper

    def encode_prompt(self, prompt: str):
        from weellm.models.transformers.ltx2_gemma_streamer import LTX2GemmaStreamer
        from transformers import AutoTokenizer

        text_enc_dir = os.path.join(self.model_dir, "text_encoder")
        ckpt_path = text_enc_dir
        tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        log.info("Loading LTX2 Gemma streamer ...")
        # Run Gemma in float32 — bfloat16 overflows at layer 6 (softmax NaN in attention).
        # Streaming one layer at a time keeps peak VRAM impact minimal (~600 MB vs ~300 MB).
        streamer = LTX2GemmaStreamer.from_pretrained(
            model_dir=ckpt_path,
            device=self.device,
            dtype=torch.float32,
            prefetch=False,
            cache_to_ram=self.cache_to_ram,
        )
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=256, # LTX uses context 256 usually
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = streamer.model(
                input_ids=text_inputs.input_ids,
                # Pass attention_mask=None to skip causal-mask creation, which triggers a
                # .item() call on meta tensors when streaming layers. As a text-encoder
                # we want bidirectional full attention anyway.
                attention_mask=None,
                output_hidden_states=True,
            )
            # LTX-2.5 connectors expect all hidden states concatenated:
            # shape: (batch, seq, hidden*num_layers) = (batch, 256, 188160)
            # The transformer's caption_projection then takes the connector output (3840-dim).
            text_encoder_hidden_states = outputs.hidden_states
            
            # Stack in float32 first to avoid bfloat16 overflow -> NaN
            stacked = torch.stack(list(text_encoder_hidden_states), dim=-1).float()  # (B, seq, hidden, layers)
            prompt_embeds = stacked.flatten(2, 3)  # (B, seq, hidden*layers)
            # Clamp to safe float32 range, then convert to target dtype
            prompt_embeds = prompt_embeds.clamp(-1e4, 1e4).to(dtype=self.torch_dtype)
            # Keep the real padding mask for connectors
            prompt_attention_mask = text_inputs.attention_mask

        del streamer, tokenizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return prompt_embeds, prompt_attention_mask

    def __call__(self, prompt: str, **kwargs):
        height = kwargs.get("height", 512)
        width = kwargs.get("width", 512)
        num_frames = kwargs.get("num_frames", 49) # LTX defaults to 121 or 49
        num_inference_steps = kwargs.get("num_inference_steps", 20)
        
        first_frame = kwargs.get("image")
        generator = kwargs.get("generator")
        
        cache_dir = DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        embeds_cache = os.path.join(cache_dir, "ltx_prompt_embeds.pt")
        mask_cache = os.path.join(cache_dir, "ltx_prompt_mask.pt")
        
        log.info(f"Target: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
        
        if os.path.exists(embeds_cache) and os.path.exists(mask_cache):
            log.info("Phase 1 cache found — loading prompt_embeds from disk ...")
            prompt_embeds = torch.load(embeds_cache, map_location="cpu")
            prompt_attention_mask = torch.load(mask_cache, map_location="cpu")
        else:
            t0 = time.time()
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt)
            log.info(f"Phase 1 (encode) done in {time.time() - t0:.1f} s")
            torch.save(prompt_embeds.cpu(), embeds_cache)
            torch.save(prompt_attention_mask.cpu(), mask_cache)

        # Build diffusers pipeline with streamers
        from weellm.models.transformers.ltx2_dit_model import LTX2DiTModelStreamer
        try:
            from diffusers import LTX2ImageToVideoPipeline as I2V_PIPE, LTX2Pipeline as T2V_PIPE
        except ImportError:
            from diffusers import LTXImageToVideoPipeline as I2V_PIPE, LTXPipeline as T2V_PIPE
            
        try:
            from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
            vae_cls = AutoencoderKLLTX2Video
        except ImportError:
            from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
            vae_cls = AutoencoderKLLTXVideo
            
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        def _vram_log(msg):
            alloc = torch.cuda.memory_allocated() / 1e9
            res = torch.cuda.memory_reserved() / 1e9
            log.info(f"    [VRAM] {msg} -> Alloc: {alloc:.3f} GB | Reserved: {res:.3f} GB")

        transformer_dir = os.path.join(self.model_dir, "transformer")
        
        log.info("Loading LTX2 DiT transformer streamer ...")
        transformer_streamer = LTX2DiTModelStreamer.from_pretrained(
            transformer_dir=transformer_dir,
            device=self.device,
            dtype=self.torch_dtype,
            prefetch=True,
            cache_to_ram=self.cache_to_ram,
            prefetch_device="cpu"
        )
        _vram_log("After transformer streamer init")

        vae_dir = os.path.join(self.model_dir, "vae")
        log.info(f"Loading VAE from {vae_dir} using {vae_cls.__name__} ...")
        # In a real implementation we would stream the VAE if it's too large,
        # but LTX VAE is usually small enough for RAM
        vae = vae_cls.from_pretrained(
            vae_dir,
            torch_dtype=self.torch_dtype
        )
        # Offload to CPU during denoise to save VRAM
        vae.to("cpu")
        _vram_log("After VAE load (to CPU)")
        
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            use_dynamic_shifting=True,
        )

        if first_frame is not None:
            pipe_cls = I2V_PIPE
        else:
            pipe_cls = T2V_PIPE
            
        try:
            from weellm.models.transformers.ltx2_connectors import LTX2ConnectorsStreamer
            connectors_dir = os.path.join(self.model_dir, "connectors")
            log.info(f"Loading connectors streamer from {connectors_dir} ...")
            connectors_streamer = LTX2ConnectorsStreamer.from_pretrained(
                transformer_dir=connectors_dir,
                device=self.device,
                dtype=self.torch_dtype,
                prefetch=True,
                cache_to_ram=self.cache_to_ram,
                prefetch_device="cpu"
            )
            connectors = connectors_streamer.model
            # Suppress to device checking for connectors since they use meta params
            connectors.to = lambda *args, **kwargs: connectors
            _vram_log("After Connectors Streamer init")
        except (ImportError, OSError):
            log.info("No connectors found or unsupported diffusers version. Skipping connectors.")
            connectors = None

        class MockConfig:
            def __init__(self):
                self.sample_rate = 16000
                self.mel_hop_length = 160
                self.mel_bins = 64
                self.latent_channels = 8

        class MockAudioVAE:
            def __init__(self, device, dtype):
                self.latents_mean = torch.zeros((1,), device=device, dtype=dtype)
                self.latents_std = torch.ones((1,), device=device, dtype=dtype)
                self.mel_compression_ratio = 4
                self.temporal_compression_ratio = 4
                self.config = MockConfig()
                self.dtype = dtype
                
            def decode(self, latents, return_dict=False):
                # Dummy audio spectrogram
                return [torch.zeros((1, 1, 1), device=latents.device, dtype=latents.dtype)]
        
        mock_audio_vae = MockAudioVAE(self.device, self.torch_dtype)
        mock_audio_vae.device = self.device
        mock_audio_vae.torch_dtype = self.torch_dtype

        pipe = pipe_cls(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer_streamer.model,
            scheduler=scheduler,
            audio_vae=mock_audio_vae,
            connectors=connectors,
            vocoder=None,
        )
        # Suppress to device checking because of meta params
        pipe.transformer.to = lambda *args, **kwargs: pipe.transformer
        
        cache_dir = DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        denoise_cache = os.path.join(cache_dir, "denoise_state_ltx2.pt")
        
        def save_step_cache(pipe, step_index, timestep, callback_kwargs):
            _vram_log(f"Inside step {step_index} callback")
            latents = callback_kwargs.get("latents")
            if latents is not None:
                step_file = os.path.join(cache_dir, f"latents_step_{step_index}.pt")
                torch.save(latents.cpu(), step_file)
                log.info(f"Saved step {step_index} latents to {step_file}")
            return callback_kwargs

        if os.path.exists(denoise_cache):
            log.info("Phase 2a cache found — loading denoised latents from disk (skipping denoise) ...")
            cached = torch.load(denoise_cache, map_location="cpu")
            latents = cached["latents"]
        else:
            _vram_log("Right before pipeline __call__")
            log.info("Running denoise loop ...")
            t0 = time.time()
            if first_frame is not None:
                result = pipe(
                    image=first_frame,
                    prompt_embeds=prompt_embeds.to(self.device),
                    prompt_attention_mask=prompt_attention_mask.to(self.device),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device) if negative_prompt_embeds is not None else None,
                    negative_prompt_attention_mask=negative_prompt_attention_mask.to(self.device) if negative_prompt_attention_mask is not None else None,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=save_step_cache
                )
            else:
                result = pipe(
                    prompt_embeds=prompt_embeds.to(self.device),
                    prompt_attention_mask=prompt_attention_mask.to(self.device),
                    negative_prompt_embeds=torch.zeros_like(prompt_embeds).to(self.device),
                    negative_prompt_attention_mask=torch.zeros_like(prompt_attention_mask).to(self.device),
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=save_step_cache
                )
                
            log.info(f"Denoise done in {time.time() - t0:.1f} s")
            latents = result.frames.cpu()
            
            # Save Phase 2a cache
            torch.save({"latents": latents}, denoise_cache)
            log.info(f"Phase 2a cache saved to: {denoise_cache}")

        # Phase 2b: Decode
        log.info("Running VAE decode (video) ...")
        # Free transformer streamer
        del transformer_streamer
        pipe.transformer = None
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        t0 = time.time()
        latents = latents.to(device=self.device, dtype=vae.dtype)
        pipe.vae.to(self.device)
        
        # Decode
        with torch.no_grad():
            video = pipe.vae.decode(latents, None, return_dict=False)[0]
            video = pipe.video_processor.postprocess_video(video, output_type="pil")
            
        log.info(f"Decode done in {time.time() - t0:.1f} s")
        
        # Return a mock output that matches expected structure
        class MockOutput:
            def __init__(self, frames):
                self.frames = frames
        return MockOutput(frames=video)
