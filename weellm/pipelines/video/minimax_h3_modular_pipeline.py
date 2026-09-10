"""
minimax_h3_modular_pipeline.py
==============================
End-to-end FL2VA (first-frame + last-frame + text → video+audio) inference
using the WeeLLM layer-streaming engines.

Can run with:
  - Full safetensors weights  (from HF Hub or local directory)
  - GGUF quantized weights    (pass transformer_gguf_path / text_encoder_gguf_path)

Pipeline:
  1. Qwen3VLStreamer  — streams 27 vision + 64 LM layers
  2. MiniMaxH3DiTModelStreamer — streams 50 transformer blocks
  3. diffusers MiniMaxH3GeneratorBlocks — resize, VAE-encode, denoise loop, decode
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("generate_video")

DTYPE = torch.bfloat16

# ── Canvas helpers ────────────────────────────────────────────────────────────
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5


def snap_frames(seconds: float) -> int:
    """Next valid frame count: 17*n + 5."""
    frames = max(17 + 5, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def _vram(device: str = "cuda") -> str:
    if not torch.cuda.is_available() or device == "cpu":
        return ""
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"VRAM {a:.2f}/{r:.2f} GB"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Encode prompt + keyframes via streaming Qwen3-VL
# ─────────────────────────────────────────────────────────────────────────────
def encode_prompt(
    prompt: str,
    first_frame,
    last_frame,
    text_enc_dir: str,
    device: str = "cuda",
    text_encoder_gguf_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (prompt_embeds, text_token_tags).
    Mirrors MiniMaxH3FL2VATextEncoderStep.__call__
    """
    from weellm.models.transformers.qwen3_vl_streamer import Qwen3VLStreamer
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    text_enc_dir = str(text_enc_dir)

    log.info("Loading Qwen3-VL streamer ...")
    if text_encoder_gguf_path is not None:
        from weellm.seeker import override_weights_path
        ctx = override_weights_path(text_encoder_gguf_path)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        streamer = Qwen3VLStreamer.from_pretrained(
            model_dir=text_enc_dir,
            device=device,
            dtype=DTYPE,
            prefetch=False,
            cache_to_ram=False,
        )

    text_encoder = streamer.model
    tokenizer = Qwen2TokenizerFast.from_pretrained(text_enc_dir)
    processor  = Qwen3VLProcessor.from_pretrained(text_enc_dir)

    # Token-type tag constants (from MiniMaxH3ModularPipeline)
    text_tag  = 1   # text rows
    video_tag = 0   # keyframe rows

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
    log.info("  Running Qwen3-VL forward (streaming 91 blocks) ... %s", _vram(device))

    with torch.no_grad():
        prompt_embeds = get_qwen3vl_prompt_embeds(
            text_encoder,
            processor,
            token_ids,
            vision_inputs,
            text_encoder_layer=50,
            device=device,
            dtype=DTYPE,
        )

    text_token_tags = torch.tensor(token_tags, dtype=torch.long)
    log.info("  prompt_embeds: %s  %s", tuple(prompt_embeds.shape), _vram(device))

    # Free text encoder memory before loading video transformer
    del streamer, text_encoder, tokenizer, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  Text encoder freed.  %s", _vram(device))

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
    model_root: str,
    transformer_dir: str,
    cache_dir: str = None,
    no_cache: bool = False,
    lora_loader=None,
    transformer_gguf_path: str | None = None,
    device: str = "cuda",
) -> dict:
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from weellm.models.transformers.minimax_h3_dit_model import MiniMaxH3DiTModelStreamer

    log.info("Loading video transformer streamer ...")
    transformer_streamer = MiniMaxH3DiTModelStreamer.from_pretrained(
        transformer_dir=transformer_dir,
        device=device,
        dtype=DTYPE,
        prefetch=True,
        prefetch_device="cpu",
        cache_to_ram=False,
        gguf_path=transformer_gguf_path,
    )

    if lora_loader is not None:
        transformer_streamer.lora_loader = lora_loader
        lora_loader.apply_to_module(transformer_streamer.model, "")

    log.info("Building pipeline with auto-routing blocks ...")

    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3AutoBeforeEncodeStep,
        MiniMaxH3AutoVaeEncoderStep,
        MiniMaxH3AutoDenoiseStep,
        MiniMaxH3DecodeStep,
    )

    class WeeLLMDenoiseBlocks(SequentialPipelineBlocks):
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
    import transformers as _transformers
    import diffusers as _diffusers
    for _cls in ("MiniMaxH3Qwen3VLHFEncoder",):
        if not hasattr(_transformers, _cls):
            setattr(_transformers, _cls, type(_cls, (), {}))
    for _cls in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE", "MiniMaxH3DiTModel"):
        if not hasattr(_diffusers, _cls):
            setattr(_diffusers, _cls, type(_cls, (), {}))

    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        model_root,
        collection="h3",
    )

    pipe.transformer = transformer_streamer.model
    pipe._blocks = denoise_blocks

    from weellm.models.vaes.minimax_vae import MiniMaxVAEStreamer

    log.info("Loading MiniMaxH3 Video VAE streamer ...")
    vae_dir = os.path.join(model_root, "vae")
    pipe.vae = MiniMaxVAEStreamer.from_pretrained(
        vae_dir=vae_dir,
        device=device,
        dtype=DTYPE,
        cache_to_ram=False,
    )

    log.info("Loading Audio VAE ...")
    audio_vae_dir = os.path.join(model_root, "audio_vae")

    MiniMaxH3ModularPipeline._execution_device = property(lambda self: torch.device(device))
    MiniMaxH3ModularPipeline.vae_frames_per_chunk = property(lambda self: 17)
    MiniMaxH3ModularPipeline.vae_latents_per_chunk = property(lambda self: 5)
    MiniMaxH3ModularPipeline.vae_latent_channels = property(lambda self: 24)
    MiniMaxH3ModularPipeline.audio_sampling_rate = property(lambda self: 32000)
    MiniMaxH3ModularPipeline.audio_latent_channels = property(lambda self: 32)
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: 0.0)

    import importlib
    import importlib.machinery
    import importlib.util
    import sys

    audio_pkg_name = "minimax_audio_vae_pkg"
    if audio_pkg_name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(audio_pkg_name, None, is_package=True)
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [str(audio_vae_dir)]
        sys.modules[audio_pkg_name] = pkg

    minimax_h3_audio_vae = importlib.import_module(f"{audio_pkg_name}.minimax_h3_audio_vae")
    MiniMaxH3AudioVAE = minimax_h3_audio_vae.MiniMaxH3AudioVAE

    pipe.audio_vae = MiniMaxH3AudioVAE.from_pretrained(audio_vae_dir).to(device=device, dtype=torch.float32)

    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    pipe.scheduler = MiniMaxH3Scheduler(shift=12.0)
    pipe.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)

    # ── PHASE 2a: DENOISE ─────────────────────────────────────────────────────
    denoise_cache = os.path.join(cache_dir, "denoise_state.pt")

    if not no_cache and os.path.exists(denoise_cache):
        log.info("Phase 2a cache found — loading denoised latents from disk ...")
        cached = torch.load(denoise_cache, map_location="cpu")
        log.info("  Loaded latents: %s  audio_latents: %s",
                 tuple(cached["latents"].shape), tuple(cached["audio_latents"].shape))
    else:
        log.info("Running denoise loop (%d steps, %d frames) ... %s", num_inference_steps, num_frames, _vram(device))

        t0 = time.time()
        with torch.no_grad():
            denoise_state = pipe(
                prompt_embeds=prompt_embeds.to(device),
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
        log.info("  Denoise done in %.1f s  %s", elapsed, _vram(device))

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
    log.info("Running VAE decode (video + audio) ... %s", _vram(device))

    del transformer_streamer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  Transformer freed.  %s", _vram(device))

    pipe._blocks = decode_blocks

    t0 = time.time()
    with torch.no_grad():
        decode_state = pipe(
            **{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in cached.items()},
        )
    elapsed = time.time() - t0
    log.info("  Decode done in %.1f s  %s", elapsed, _vram(device))

    return decode_state


# ─────────────────────────────────────────────────────────────────────────────
# Class Wrapper
# ─────────────────────────────────────────────────────────────────────────────
from weellm.weevideopipeline import WeeVideoPipeline


class WeeMiniMaxPipeline(WeeVideoPipeline):
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        vae_tile_size: int = 256,
        transformer_gguf_path: str | None = None,
        text_encoder_gguf_path: str | None = None,
        **kwargs,
    ) -> "WeeMiniMaxPipeline":
        """
        Create a WeeMiniMaxPipeline.

        Args:
            model_dir: HuggingFace repo ID or local path to the FL2VA pipeline root.
                       e.g. "MiniMaxAI/MiniMax-H3-FL2VA" or "/kaggle/working/MiniMax-H3-FL2VA"
            transformer_gguf_path: Optional path to a GGUF file for the transformer.
                       e.g. "unsloth/MiniMax-H3-GGUF/minimax_h3_fl2va_pruned-Q4_K.gguf"
                       Can be a HuggingFace Hub path (org/repo/filename) or local.
            text_encoder_gguf_path: Optional GGUF path for the Qwen3-VL text encoder.
        """
        wrapper = cls.__new__(cls)
        object.__setattr__(wrapper, "_pipeline", None)
        object.__setattr__(wrapper, "model_dir", str(model_dir))
        object.__setattr__(wrapper, "device", device)
        object.__setattr__(wrapper, "torch_dtype", torch_dtype)
        object.__setattr__(wrapper, "cache_to_ram", cache_to_ram)
        object.__setattr__(wrapper, "transformer_gguf_path", transformer_gguf_path)
        object.__setattr__(wrapper, "text_encoder_gguf_path", text_encoder_gguf_path)

        try:
            from weellm.models.loras.lora_loader import MiniMaxH3LoRALoader
            lora_loader = MiniMaxH3LoRALoader()
            object.__setattr__(wrapper, "lora_loader", lora_loader)
        except Exception as e:
            log.warning(f"Failed to load LoRA: {e}")
            object.__setattr__(wrapper, "lora_loader", None)

        return wrapper

    def _resolve_model_root(self) -> str:
        """
        Resolve model_dir to a local path, downloading from HF Hub if needed.
        Returns the local path of the pipeline root directory.
        """
        model_dir = self.model_dir
        p = Path(model_dir)
        if p.exists():
            return str(p)

        # It's a HF Hub repo ID — download config files + model_index only
        # (actual weights will be streamed on demand by the seekers)
        parts = model_dir.replace("\\", "/").split("/")
        if len(parts) >= 2:
            from huggingface_hub import snapshot_download
            log.info("Downloading pipeline config from HF Hub: %s ...", model_dir)
            local = snapshot_download(
                repo_id=model_dir,
                allow_patterns=[
                    "*.json",
                    "vae/*.json",
                    "audio_vae/**",
                    "transformer/*.json",
                    "text_encoder/*.json",
                    "text_encoder/*.tiktoken",
                    "text_encoder/tokenizer*",
                    "scheduler/*.json",
                ],
            )
            return local
        return model_dir

    def __call__(self, prompt: str, **kwargs):
        height = kwargs.get("height")
        width = kwargs.get("width")
        first_frame = kwargs.get("image")

        # ── MiniMax Canvas Auto-Resolution ──────────────────────────────────────
        if first_frame is not None and (height is None or width is None):
            try:
                aspect = first_frame.width / first_frame.height
                canvases = [
                    (544, 960), (576, 1024), (640, 1152), (704, 1280), (768, 1344),
                    (960, 544), (1152, 640), (1344, 768),
                    (544, 544), (768, 768),
                    (576, 768), (768, 1024),
                    (768, 576), (1024, 768),
                    (512, 1152), (672, 1536),
                ]
                fastest = {}
                for h, w in canvases:
                    r = w / h
                    if r not in fastest or w * h < fastest[r][0] * fastest[r][1]:
                        fastest[r] = (h, w)
                ratio = min(fastest, key=lambda r: abs(r - aspect))
                target_h, target_w = fastest[ratio]
                if width is None: width = target_w
                if height is None: height = target_h
            except Exception:
                if width is None: width = 960
                if height is None: height = 544
        else:
            if width is None: width = 960
            if height is None: height = 544

        num_frames = kwargs.get("num_frames", 75)
        num_frames = snap_frames(num_frames / FPS)

        num_inference_steps = kwargs.get("num_inference_steps", 6)
        last_frame = kwargs.get("last_image")

        generator = kwargs.get("generator")
        seed = generator.initial_seed() if generator is not None else 42

        device = self.device

        # ── Resolve model root ─────────────────────────────────────────────────
        model_root = self._resolve_model_root()
        text_enc_dir = os.path.join(model_root, "text_encoder")
        transformer_dir = os.path.join(model_root, "transformer")

        # ── Set up VideoStepCache via Base Class ─────────────────────────────
        cache = self._setup_cache(prompt, height, width, num_frames, num_inference_steps, seed)
        cache_dir = cache.run_dir_path

        log.info("Target: %dx%d, %d frames (%.2fs), %d steps", width, height, num_frames, num_frames / FPS, num_inference_steps)

        # ── PHASE 1: Text encode ─────────────────────────────────────────────────
        cached_embeds = cache.load_embeds()
        if cached_embeds is not None:
            prompt_embeds   = cached_embeds["prompt_embeds"]
            text_token_tags = cached_embeds["text_token_tags"]
        else:
            t0 = time.time()
            prompt_embeds, text_token_tags = encode_prompt(
                prompt,
                first_frame,
                last_frame,
                text_enc_dir=text_enc_dir,
                device=device,
                text_encoder_gguf_path=getattr(self, "text_encoder_gguf_path", None),
            )
            log.info("Phase 1 (encode) done in %.1f s", time.time() - t0)
            cache.save_embeds({
                "prompt_embeds":   prompt_embeds.cpu(),
                "text_token_tags": text_token_tags.cpu(),
            })

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
            model_root=model_root,
            transformer_dir=transformer_dir,
            cache_dir=cache_dir,
            no_cache=False,
            lora_loader=getattr(self, "lora_loader", None),
            transformer_gguf_path=getattr(self, "transformer_gguf_path", None),
            device=device,
        )
        log.info("Phase 2 (denoise+decode) done in %.1f s", time.time() - t0)

        # ── Denormalize MiniMax output (ImageNet mean/std) ────────────────────
        videos = decode_state.get("videos")
        if isinstance(videos, torch.Tensor):
            mean = torch.tensor([0.485, 0.456, 0.406], device=videos.device, dtype=videos.dtype).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=videos.device, dtype=videos.dtype).view(-1, 1, 1)

            if videos.shape[0] == 3 and len(videos.shape) == 4:
                mean = mean.unsqueeze(1)
                std = std.unsqueeze(1)
            elif len(videos.shape) == 4 and videos.shape[1] == 3:
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)

            videos = (videos * std) + mean
            decode_state["videos"] = videos.clamp(0, 1)

        decode_state["fps"] = 24

        class MiniMaxOutput:
            def __init__(self, state):
                self.videos = state.get("videos")
                self.audio = state.get("audio")
                self.fps = state.get("fps")
                self.sampling_rate = 24000

        return MiniMaxOutput(decode_state)
