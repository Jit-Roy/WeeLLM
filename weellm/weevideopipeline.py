"""
generate_video.py
=================
End-to-end fl2va (first-frame + last-frame + text → video+audio) inference
using the weellm layer-streaming engines on an RTX 3050 4GB laptop.

Pipeline:
  1. Qwen3VLStreamer  — streams 27 vision + 64 LM layers, reads hidden_states[50]
  2. MiniMaxH3DiTModelStreamer — streams 50 transformer blocks
  3. diffusers MiniMaxH3GeneratorBlocks — resize, VAE-encode, denoise loop, decode

Usage:
  python generate_video.py --prompt "A cat walks across a wooden floor" \\
                           --first_frame first.jpg \\
                           --last_frame last.jpg \\
                           --output out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import logging
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("generate_video")

import torch
from PIL import Image, ImageOps

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_ROOT  = r"D:\Personal Projects\LightLLM\MiniMax-H3\FL2VA"
TEXT_ENC_DIR = os.path.join(MODEL_ROOT, "text_encoder")
TRANSFORMER_DIR = os.path.join(MODEL_ROOT, "transformer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

TEXT_ENCODER_LAYER = 50   # MiniMax-H3 reads hidden_states[50] of Qwen3-VL

# ── Cache directory (intermediate results so crashes don't cost you a re-run) ─
DEFAULT_CACHE_DIR = r"D:\Personal Projects\LightLLM\.weellm_cache"

# ── Canvas helpers ────────────────────────────────────────────────────────────
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5

def snap_frames(seconds: float) -> int:
    """Next valid frame count: 17*n + 5."""
    frames = max(17 + 5, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def _vram() -> str:
    if DEVICE != "cuda":
        return ""
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"VRAM {a:.2f}/{r:.2f} GB"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Encode prompt + keyframes via streaming Qwen3-VL
# ─────────────────────────────────────────────────────────────────────────────
def encode_prompt(
    prompt: str,
    first_frame: Image.Image | None,
    last_frame: Image.Image | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (prompt_embeds, text_token_tags).
    Mirrors MiniMaxH3FL2VATextEncoderStep.__call__
    """
    from weellm.models.transformers.qwen3_vl_streamer import Qwen3VLStreamer
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    log.info("Loading Qwen3-VL streamer ...")
    streamer = Qwen3VLStreamer.from_pretrained(
        model_dir=TEXT_ENC_DIR,
        device=DEVICE,
        dtype=DTYPE,
        prefetch=False,
        cache_to_ram=False,
    )
    text_encoder = streamer.model
    tokenizer = Qwen2TokenizerFast.from_pretrained(TEXT_ENC_DIR)
    processor  = Qwen3VLProcessor.from_pretrained(TEXT_ENC_DIR)

    # Token-type tag constants (from MiniMaxH3ModularPipeline)
    text_tag  = 1   # text rows
    video_tag = 0   # keyframe rows (confusingly called 'video' in the pipeline)

    # ── Build presentation ───────────────────────────────────────────────────
    keyframes = []
    if first_frame is not None:
        keyframes.append(first_frame)
    if last_frame is not None:
        keyframes.append(last_frame)

    vision_inputs: dict = {}
    image_grid_thw = None
    if keyframes:
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        vision_inputs = {
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": image_grid_thw,
        }

    token_ids:  list[int] = []
    token_tags: list[int] = []
    if keyframes:
        merge_size = processor.image_processor.merge_size ** 2
        for idx in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[idx].prod()) // merge_size
            label_ids = tokenizer(f"<Picture {idx + 1}>: ", add_special_tokens=False)["input_ids"]
            vis_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids  += label_ids + vis_ids
            token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vis_ids)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids  += prompt_ids
    token_tags += [text_tag] * len(prompt_ids)

    log.info("  Presentation: %d tokens (%d keyframes + prompt)", len(token_ids), len(keyframes))
    log.info("  Running Qwen3-VL forward (streaming 91 blocks) ... %s", _vram())

    with torch.no_grad():
        prompt_embeds = get_qwen3vl_prompt_embeds(
            text_encoder,
            processor,
            token_ids,
            vision_inputs,
            text_encoder_layer=TEXT_ENCODER_LAYER,
            device=DEVICE,
            dtype=DTYPE,
        )

    text_token_tags = torch.tensor(token_tags, dtype=torch.long)
    log.info("  prompt_embeds: %s  %s", tuple(prompt_embeds.shape), _vram())

    # Free text encoder memory before loading video transformer
    del streamer, text_encoder, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    log.info("  Text encoder freed.  %s", _vram())

    return prompt_embeds, text_token_tags


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Denoise + decode via streaming video transformer + diffusers pipeline
# ─────────────────────────────────────────────────────────────────────────────
def denoise_and_decode(
    prompt_embeds: torch.Tensor,
    text_token_tags: torch.Tensor,
    first_frame,
    last_frame,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
    no_cache: bool = False,
    lora_loader = None,
) -> dict:
    from diffusers.utils import encode_video
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from weellm.models.transformers.minimax_h3_dit_model import MiniMaxH3DiTModelStreamer

    log.info("Loading video transformer streamer ...")
    transformer_streamer = MiniMaxH3DiTModelStreamer.from_pretrained(
        transformer_dir=TRANSFORMER_DIR,
        device=DEVICE,
        dtype=DTYPE,
        prefetch=True,
        prefetch_device="cpu",   # load next block to RAM while GPU computes — no VRAM cost
        cache_to_ram=False,
    )
    
    if lora_loader is not None:
        transformer_streamer.lora_loader = lora_loader
        # Apply LoRA patches to the resident keys (e.g., norm_out) on the base model!
        lora_loader.apply_to_module(transformer_streamer.model, "")

    log.info("Building pipeline with auto-routing blocks (skips text_encoder, uses conditional vae+denoise) ...")

    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3AutoBeforeEncodeStep,   # Handles resize for fl2va or no-op for t2va
        MiniMaxH3AutoVaeEncoderStep,     # fl2va → keyframe encode, t2va → no-op
        MiniMaxH3AutoDenoiseStep,        # fl2va → FL2VACoreDenoiseStep, t2va → CoreDenoiseStep
        MiniMaxH3DecodeStep,             # Always: video + audio decode
    )

    class WeeLLMDenoiseBlocks(SequentialPipelineBlocks):
        """Phase 2a: before_encode + vae_encoder + denoise (no VAE decode)."""
        model_name = "minimax-h3"
        block_classes = [
            MiniMaxH3AutoBeforeEncodeStep,
            MiniMaxH3AutoVaeEncoderStep,
            MiniMaxH3AutoDenoiseStep,
        ]
        block_names = ["before_encode", "vae_encoder", "denoise"]

        @property
        def outputs(self):
            from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam
            return [
                OutputParam("latents", type_hint=torch.Tensor),
                OutputParam("audio_latents", type_hint=torch.Tensor),
            ]

    class WeeLLMDecodeBlocks(SequentialPipelineBlocks):
        """Phase 2b: decode denoised latents → video + audio."""
        model_name = "minimax-h3"
        block_classes = [MiniMaxH3DecodeStep]
        block_names = ["decode"]

        @property
        def inputs(self):
            from diffusers.modular_pipelines.modular_pipeline_utils import InputParam
            return [
                InputParam("latents",                    type_hint=torch.Tensor, required=True),
                InputParam("audio_latents",              type_hint=torch.Tensor, required=True),
                InputParam("num_condition_video_rows",   type_hint=int, default=0),
                InputParam("num_condition_audio_rows",   type_hint=int, default=0),
                InputParam("num_latent_frames",          type_hint=int, required=True),
                InputParam("latent_height",              type_hint=int, required=True),
                InputParam("latent_width",               type_hint=int, required=True),
                InputParam("num_audio_latents",          type_hint=int, required=True),
                InputParam("output_type",                type_hint=str, default="pil"),
            ]

        @property
        def outputs(self):
            from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam
            return [
                OutputParam("videos", type_hint=list),
                OutputParam("audio",  type_hint=list),
                OutputParam("sampling_rate", type_hint=int),
            ]

    denoise_blocks = WeeLLMDenoiseBlocks()
    decode_blocks  = WeeLLMDecodeBlocks()

    # ── Stub injection ────────────────────────────────────────────────────────
    # model_index.json references several MiniMax-internal classes that don't ship
    # in the public transformers/diffusers packages. Inject dummy stubs so
    # from_pretrained can parse the config without crashing. We override all real
    # components (transformer, vae, audio_vae) immediately after, so stubs are never used.
    import transformers as _transformers
    import diffusers as _diffusers
    for _cls in ("MiniMaxH3Qwen3VLHFEncoder",):
        if not hasattr(_transformers, _cls):
            setattr(_transformers, _cls, type(_cls, (), {}))
    for _cls in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE", "MiniMaxH3DiTModel"):
        if not hasattr(_diffusers, _cls):
            setattr(_diffusers, _cls, type(_cls, (), {}))

    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        MODEL_ROOT,
        collection="h3",
    )
    
    # Set the streaming transformer explicitly — from_pretrained ignores it as a kwarg
    pipe.transformer = transformer_streamer.model
    
    # Override the default blocks with our denoise blocks (decode runs separately)
    pipe._blocks = denoise_blocks
    
    from weellm.models.vaes.minimax_vae import MiniMaxVAEStreamer
    
    log.info("Loading MiniMaxH3 Video VAE streamer (10GB block streaming) ...")
    vae_dir = os.path.join(MODEL_ROOT, "vae")
    pipe.vae = MiniMaxVAEStreamer.from_pretrained(
        vae_dir=vae_dir,
        device=DEVICE,
        dtype=DTYPE,
        cache_to_ram=False,
    )
    
    log.info("Loading Audio VAE ...")
    audio_vae_dir = os.path.join(MODEL_ROOT, "audio_vae")
    
    # Diffusers pipeline attributes expected by modular blocks — must be set before
    # ANY pipe() call (denoise or decode). We patch the class so all instances share them.
    MiniMaxH3ModularPipeline._execution_device = property(lambda self: torch.device(DEVICE))
    MiniMaxH3ModularPipeline.vae_frames_per_chunk = property(lambda self: 17)
    MiniMaxH3ModularPipeline.vae_latents_per_chunk = property(lambda self: 5)
    MiniMaxH3ModularPipeline.vae_latent_channels = property(lambda self: 24)
    MiniMaxH3ModularPipeline.audio_sampling_rate = property(lambda self: 32000)
    MiniMaxH3ModularPipeline.audio_latent_channels = property(lambda self: 32)
    # Bypass minimum 5.0s assertion for fast testing
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: 0.0)

    import importlib.util
    import importlib.machinery
    import importlib
    import sys
    
    audio_pkg_name = "minimax_audio_vae_pkg"
    if audio_pkg_name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(audio_pkg_name, None, is_package=True)
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [str(audio_vae_dir)]
        sys.modules[audio_pkg_name] = pkg

    minimax_h3_audio_vae = importlib.import_module(f"{audio_pkg_name}.minimax_h3_audio_vae")
    MiniMaxH3AudioVAE = minimax_h3_audio_vae.MiniMaxH3AudioVAE
    
    pipe.audio_vae = MiniMaxH3AudioVAE.from_pretrained(audio_vae_dir).to(device=DEVICE, dtype=torch.float32)
    # MiniMaxH3Scheduler is the proper scheduler — video uses shift=12.0, audio shift=3.0
    # (from model_index.json sigma_shift_scales)
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    pipe.scheduler = MiniMaxH3Scheduler(shift=12.0)
    pipe.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)

    # ── PHASE 2a: DENOISE ─────────────────────────────────────────────────────
    denoise_cache = os.path.join(cache_dir, "denoise_state.pt")

    if not no_cache and os.path.exists(denoise_cache):
        log.info("Phase 2a cache found — loading denoised latents from disk (skipping denoise) ...")
        cached = torch.load(denoise_cache, map_location="cpu")
        log.info("  Loaded latents: %s  audio_latents: %s",
                 tuple(cached["latents"].shape), tuple(cached["audio_latents"].shape))
    else:
        log.info("Running denoise loop (%d steps, %d frames) ... %s", num_inference_steps, num_frames, _vram())

        t0 = time.time()
        with torch.no_grad():
            denoise_state = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE),
                text_token_tags=text_token_tags,
                image=first_frame,
                last_image=last_frame,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
            )
        elapsed = time.time() - t0
        log.info("  Denoise done in %.1f s  %s", elapsed, _vram())

        # Save all state tensors to cache so a decode crash doesn't cost a re-denoise
        cached = {}
        for key in ["latents", "audio_latents", "num_condition_video_rows",
                    "num_condition_audio_rows", "num_latent_frames", "latent_height",
                    "latent_width", "num_audio_latents"]:
            val = denoise_state.get(key)
            if val is not None:
                cached[key] = val.cpu() if isinstance(val, torch.Tensor) else val
        torch.save(cached, denoise_cache)
        log.info("  Phase 2a cache saved to: %s", denoise_cache)

    # ── PHASE 2b: DECODE ──────────────────────────────────────────────────────
    log.info("Running VAE decode (video + audio) ... %s", _vram())

    # Free the transformer streamer — we don't need it for decode
    del transformer_streamer
    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    log.info("  Transformer freed.  %s", _vram())

    # Switch pipe to decode blocks
    pipe._blocks = decode_blocks

    t0 = time.time()
    with torch.no_grad():
        decode_state = pipe(
            **{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
               for k, v in cached.items()},
        )
    elapsed = time.time() - t0
    log.info("  Decode done in %.1f s  %s", elapsed, _vram())

    return decode_state


# ─────────────────────────────────────────────────────────────────────────────
# Class Wrapper
# ─────────────────────────────────────────────────────────────────────────────
from weellm.pipeline import WeeBasePipeline

class WeeVideoPipeline(WeeBasePipeline):
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
    ) -> WeeVideoPipeline:
        wrapper = cls.__new__(cls)
        object.__setattr__(wrapper, "_pipeline", None)
        object.__setattr__(wrapper, "model_dir", str(model_dir))
        object.__setattr__(wrapper, "device", device)
        object.__setattr__(wrapper, "torch_dtype", torch_dtype)
        object.__setattr__(wrapper, "cache_to_ram", cache_to_ram)
        
        try:
            from weellm.models.loras.lora_loader import MiniMaxH3LoRALoader
            lora_loader = MiniMaxH3LoRALoader()
            object.__setattr__(wrapper, "lora_loader", lora_loader)
        except Exception as e:
            log.warning(f"Failed to load LoRA: {e}")
            object.__setattr__(wrapper, "lora_loader", None)
            
        return wrapper

    def __call__(self, prompt: str, **kwargs):
        height = kwargs.get("height", 544)
        width = kwargs.get("width", 960)
        num_frames = kwargs.get("num_frames", 75)
        num_frames = snap_frames(num_frames / FPS)
        
        num_inference_steps = kwargs.get("num_inference_steps", 6)
        
        first_frame = kwargs.get("image")
        last_frame = kwargs.get("last_image")
        
        generator = kwargs.get("generator")
        seed = generator.initial_seed() if generator is not None else 42

        cache_dir = DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        embeds_cache = os.path.join(cache_dir, "prompt_embeds.pt")
        tags_cache = os.path.join(cache_dir, "text_token_tags.pt")
        
        log.info("Target: %dx%d, %d frames (%.2fs), %d steps", width, height, num_frames, num_frames / FPS, num_inference_steps)
        
        # ── PHASE 1: Text encode ─────────────────────────────────────────────────
        if os.path.exists(embeds_cache) and os.path.exists(tags_cache):
            log.info("Phase 1 cache found — loading prompt_embeds from disk (skipping Qwen3-VL) ...")
            prompt_embeds = torch.load(embeds_cache, map_location="cpu")
            text_token_tags = torch.load(tags_cache, map_location="cpu")
        else:
            t0 = time.time()
            prompt_embeds, text_token_tags = encode_prompt(prompt, first_frame, last_frame)
            log.info("Phase 1 (encode) done in %.1f s", time.time() - t0)
            torch.save(prompt_embeds.cpu(), embeds_cache)
            torch.save(text_token_tags.cpu(), tags_cache)

        # ── PHASE 2: Denoise + decode ─────────────────────────────────────────────
        t0 = time.time()
        decode_state = denoise_and_decode(
            prompt_embeds,
            text_token_tags,
            first_frame=first_frame,
            last_frame=last_frame,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            cache_dir=cache_dir,
            no_cache=False,
            lora_loader=getattr(self, "lora_loader", None)
        )
        log.info("Phase 2 (denoise+decode) done in %.1f s", time.time() - t0)
        
        return decode_state
