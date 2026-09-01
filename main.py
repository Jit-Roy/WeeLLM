"""
main.py -- WeeLLM: Layer-streaming inference for large diffusion models.

Fits multi-billion-parameter diffusion models under 4 GB VRAM and 8 GB RAM
without quantization or model reduction.

Usage
-----
    python main.py --model black-forest-labs/FLUX.1-dev --prompt "A sunset over mountains"
    python main.py --model Tongyi-MAI/Z-Image-Turbo --height 768 --width 768 --steps 4 --seed 42
    python main.py --help
"""

import os

# Fix OpenMP duplicate lib error on some platforms (e.g. Windows/Conda)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Configure the weellm logger based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=level,
    )
    logging.getLogger("weellm").setLevel(level)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weellm",
        description=(
            "WeeLLM — Layer-streaming diffusion inference.\n"
            "Runs large models under 4 GB VRAM without quantization.\n"
            "Supports local directories or Hugging Face repository IDs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            examples:
            python main.py --model black-forest-labs/FLUX.1-dev --prompt "A majestic lion at golden hour"
            python main.py --model Tongyi-MAI/Z-Image-Turbo --height 768 --width 768 --steps 4
            python main.py --model ./my-local-flux-model --no_prefetch          # lower RAM usage
            python main.py --model runwayml/stable-diffusion-v1-5 --prompt "..." --negative_prompt "blurry"
            python main.py --model black-forest-labs/FLUX.1-schnell --dry_run   # load only, no generation
        """,
    )

    # Text-to-Image 
    parser.add_argument(
        "--model", type=str, required=True,
        metavar="ID_OR_PATH",
        help="Hugging Face repo ID or path to local model directory (e.g. Tongyi-MAI/Z-Image-Turbo)",
    )
    parser.add_argument(
        "--transformer", type=str, default=None,
        help="Optional path to a custom transformer/UNet weights file (e.g., a .gguf file). Overrides the base model's transformer."
    )
    parser.add_argument(
        "--text_encoder", type=str, default=None,
        help="Optional path to a custom text_encoder weights file (e.g., a .gguf file). Overrides the base model's text_encoder."
    )
    parser.add_argument(
        "--text_encoder_2", type=str, default=None,
        help="Optional path to a custom text_encoder_2 weights file (e.g., a .gguf file). Overrides the base model's text_encoder_2."
    )
    parser.add_argument(
        "--text_encoder_3", type=str, default=None,
        help="Optional path to a custom text_encoder_3 weights file (e.g., a .gguf file). Overrides the base model's text_encoder_3."
    )
    parser.add_argument(
        "--text_encoder_4", type=str, default=None,
        help="Optional path to a custom text_encoder_4 weights file (e.g., a .gguf file). Overrides the base model's text_encoder_4."
    )
    parser.add_argument(
        "--prompt", type=str,
        default="A majestic lion in the savanna at golden hour",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="Negative prompt to guide generation away from unwanted content (default: empty)",
    )
    parser.add_argument("--height", type=int, default=None, help="Output height in pixels (default: 1024 for T2I, original for I2I)")
    parser.add_argument("--width",  type=int, default=None, help="Output width in pixels (default: 1024 for T2I, original for I2I)")
    parser.add_argument("--steps",  type=int, default=4,   help="Denoising steps         (default: 4)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of video frames (default: 75 for video models)")
    parser.add_argument(
        "--guidance_scale", type=float, default=None,
        help="Classifier-free guidance scale (default: use pipeline default, usually 4.5 for SD3/LongCat or 1.0/3.5 for Flux)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (default: -1 for random)")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype (default: bfloat16)",
    )
    
    # Image-to-Image / Editing
    parser.add_argument(
        "--image", type=str, default="",
        help="Path to an input image (or first frame for video) for image-to-image or editing tasks",
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=None,
        help="Image guidance scale for editing models like HiDream (default: use pipeline default)",
    )
    parser.add_argument(
        "--mask_image", type=str, default="",
        help="Path to a mask image for inpainting tasks",
    )
    parser.add_argument(
        "--strength", type=float, default=0.8,
        help="Denoising strength for image-to-image (default: 0.8)",
    )

    # Video generation
    parser.add_argument(
        "--last_image", type=str, default="",
        help="Path to the last frame image (specifically for MiniMax-H3 FL2VA video generation)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="output.png",
        help="Output image file path (default: output.png)",
    )

    # Advanced / performance
    parser.add_argument(
        "--no_prefetch", action="store_true",
        help="Disable background prefetching (slower but uses less RAM)",
    )
    parser.add_argument(
        "--cache_to_ram", action="store_true",
        help="Load safetensors into CPU RAM instead of streaming from disk (faster on Kaggle/Colab)",
    )
    parser.add_argument(
        "--vram_budget", type=float, default=None,
        help="VRAM budget in GB. If None, dynamically calculates based on hardware.",
    )
    parser.add_argument(
        "--ram_budget", type=float, default=None,
        help="CPU RAM budget in GB. If None, dynamically calculates based on hardware.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load the pipeline and verify setup without running inference. Useful for CI/testing.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug logging (including per-layer VRAM tracking).",
    )
    parser.add_argument(
        "--vae_tile_size", type=int, default=512,
        help="Tile size for VAE decoding to prevent VRAM spikes. 256=low VRAM but moire artifacts, 512=default, 1024=high VRAM.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # Automatically disable prefetch for GGUF models to prevent GPU contention during dequantization
    is_gguf = (args.transformer and str(args.transformer).lower().endswith('.gguf')) or \
              (args.text_encoder and str(args.text_encoder).lower().endswith('.gguf')) or \
              (args.text_encoder_2 and str(args.text_encoder_2).lower().endswith('.gguf')) or \
              (args.text_encoder_3 and str(args.text_encoder_3).lower().endswith('.gguf')) or \
              (args.text_encoder_4 and str(args.text_encoder_4).lower().endswith('.gguf')) or \
              (args.model and str(args.model).lower().endswith('.gguf'))
    if is_gguf and not args.no_prefetch:
        print("\n[WeeLLM] Auto-routing to --no_prefetch mode because GGUF models require GPU dequantization.")
        args.no_prefetch = True

    _model_lower = args.model.lower()
    _is_minimax_default = "minimax" in _model_lower or "fl2va" in _model_lower or "h3" in _model_lower

    if _is_minimax_default:
        if args.image:
            try:
                from PIL import Image
                with Image.open(args.image) as img:
                    aspect = img.width / img.height
                
                canvases = [
                    (544, 960), (576, 1024), (640, 1152), (704, 1280), (768, 1344), # 16:9
                    (960, 544), (1152, 640), (1344, 768), # 9:16
                    (544, 544), (768, 768), # 1:1
                    (576, 768), (768, 1024), # 4:3
                    (768, 576), (1024, 768), # 3:4
                    (512, 1152), (672, 1536), # 21:9
                ]
                
                fastest = {}
                for h, w in canvases:
                    r = w / h
                    if r not in fastest or w * h < fastest[r][0] * fastest[r][1]:
                        fastest[r] = (h, w)
                
                ratio = min(fastest, key=lambda r: abs(r - aspect))
                target_h, target_w = fastest[ratio]
                
                if args.width is None: args.width = target_w
                if args.height is None: args.height = target_h
            except Exception:
                if args.width is None: args.width = 960
                if args.height is None: args.height = 544
        else:
            if args.width is None: args.width = 960
            if args.height is None: args.height = 544
    elif args.image:
        try:
            from PIL import Image
            with Image.open(args.image) as temp_img:
                if args.width is None:
                    args.width = temp_img.width
                if args.height is None:
                    args.height = temp_img.height
        except Exception:
            if args.width is None: args.width = 1024
            if args.height is None: args.height = 1024
    else:
        if args.width  is None: args.width  = 1024
        if args.height is None: args.height = 1024

    is_flux_fill = False
    try:
        from pathlib import Path
        import json
        if Path(args.model).is_dir():
            idx_path = Path(args.model) / "model_index.json"
            if idx_path.exists():
                with open(idx_path) as f:
                    is_flux_fill = json.load(f).get("_class_name", "") == "FluxFillPipeline"
    except:
        pass

    if is_flux_fill or ("flux" in _model_lower and "fill" in _model_lower):
        args.width = max(16, (args.width // 16) * 16)
        args.height = max(16, (args.height // 16) * 16)

    _configure_logging(args.verbose)
    logger = logging.getLogger("weellm")

    # ── Device & dtype ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # ── Header ──────────────────────────────────────────────────────────────
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("  WeeLLM — Layer-Streaming Diffusion Inference")
    logger.info(sep)
    logger.info("  Model:    %s", args.model)
    logger.info("  Prompt:   %s", args.prompt)
    if args.negative_prompt:
        logger.info("  Neg:      %s", args.negative_prompt)
    logger.info("  Size:     %d x %d px", args.width, args.height)
    logger.info(
        "  Steps:    %d  |  Guidance: %s  |  Seed: %s",
        args.steps, args.guidance_scale, args.seed,
    )
    logger.info("  Dtype:    %s  |  Device: %s", args.dtype, device)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("  GPU:      %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
    if args.dry_run:
        logger.info("  Mode:     DRY RUN (no inference)")

    logger.info("")

    # ── Load pipeline ────────────────────────────────────────────────────────
    is_minimax = False
    model_index_path = None
    try:
        from pathlib import Path
        if Path(args.model).is_dir():
            model_index_path = Path(args.model) / "model_index.json"
        elif args.model.startswith("MiniMax"):
            model_index_path = None
    except:
        pass
    
    if model_index_path and model_index_path.exists():
        try:
            import json
            with open(model_index_path) as f:
                index = json.load(f)
            is_minimax = "MiniMaxH3" in index.get("_class_name", "")
        except:
            pass
            
    is_minimax = is_minimax or _is_minimax_default
    _is_ltx = "ltx" in _model_lower
    
    input_image = None
    if args.image:
        from PIL import Image, ImageOps
        try:
            input_image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
            input_image = input_image.resize((args.width, args.height), Image.LANCZOS)
        except Exception as e:
            print(f"ERROR: Could not load input image: {e}", file=sys.stderr)
            return 1

    if is_minimax:
        from weellm.weevideopipeline import WeeVideoPipeline as PipelineClass
        logger.info("  Mode:     Text-to-Video+Audio (MiniMax-H3)%s", " (With Start Image)" if input_image else "")
    elif _is_ltx:
        from weellm.weeltx2pipeline import WeeLTX2Pipeline as PipelineClass
        logger.info("  Mode:     Text-to-Video (LTX-2.5)%s", " (With Start Image)" if input_image else "")
    elif input_image:
        from weellm import WeeImagePipeline as PipelineClass
        logger.info("  Mode:     Image-to-Image / Edit (Input: %s)", args.image)
    else:
        from weellm import WeePipeline as PipelineClass
        logger.info("  Mode:     Text-to-Image")

    last_image = None
    if getattr(args, "last_image", None):
        try:
            from PIL import Image, ImageOps
            last_image = ImageOps.exif_transpose(Image.open(args.last_image)).convert("RGB")
            last_image = last_image.resize((args.width, args.height), Image.LANCZOS)
            logger.info("  Last Frame: %s", args.last_image)
        except Exception as e:
            print(f"ERROR: Could not load last_image: {e}", file=sys.stderr)
            return 1

    mask_image = None
    if getattr(args, "mask_image", None):
        try:
            from PIL import Image, ImageOps
            mask_image = ImageOps.exif_transpose(Image.open(args.mask_image)).convert("RGB")
            mask_image = mask_image.resize((args.width, args.height), Image.LANCZOS)
            logger.info("  Mask Image: %s", args.mask_image)
        except Exception as e:
            print(f"ERROR: Could not load mask_image: {e}", file=sys.stderr)
            return 1

    t_load = time.time()
    try:
        kwargs = {}
        if args.vram_budget is not None:
            kwargs["vram_budget_gb"] = args.vram_budget
        if args.ram_budget is not None:
            kwargs["ram_budget_gb"] = args.ram_budget

        if args.transformer is not None:
            kwargs["transformer_dir"] = args.transformer
        if args.text_encoder is not None:
            kwargs["text_encoder_dir"] = args.text_encoder
        if args.text_encoder_2 is not None:
            kwargs["text_encoder_2_dir"] = args.text_encoder_2
        if args.text_encoder_3 is not None:
            kwargs["text_encoder_3_dir"] = args.text_encoder_3
        if args.text_encoder_4 is not None:
            kwargs["text_encoder_4_dir"] = args.text_encoder_4

        pipe = PipelineClass.from_pretrained(
            model_dir=args.model,
            device=device,
            torch_dtype=dtype,
            prefetch=not args.no_prefetch,
            cache_to_ram=args.cache_to_ram,
            vae_tile_size=getattr(args, "vae_tile_size", 256),
            **kwargs
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    logger.info("Pipeline loaded in %.1fs\n", time.time() - t_load)

    if args.dry_run:
        logger.info("Dry run complete. Pipeline loaded successfully. Skipping inference.")
        return 0

    # ── Generate ─────────────────────────────────────────────────────────────
    t_gen = time.time()
    seed  = args.seed if args.seed != -1 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    logger.info("  Using seed: %d", seed)

    # Build call kwargs — only pass negative_prompt if the pipeline supports it
    call_kwargs: dict = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
    )
    if args.num_frames is not None:
        call_kwargs["num_frames"] = args.num_frames
    if args.guidance_scale is not None:
        call_kwargs["guidance_scale"] = args.guidance_scale
    if hasattr(args, "image_guidance_scale") and args.image_guidance_scale is not None:
        call_kwargs["image_guidance_scale"] = args.image_guidance_scale

    if input_image is not None:
        call_kwargs["image"] = input_image
        call_kwargs["strength"] = args.strength
        
    if last_image is not None:
        call_kwargs["last_image"] = last_image
        
    if mask_image is not None:
        call_kwargs["mask_image"] = mask_image
        
    if args.negative_prompt:
        call_kwargs["negative_prompt"] = args.negative_prompt

    out   = pipe(**call_kwargs)
    
    # Handle different output types (images, video+audio, etc)
    if hasattr(out, "videos") or hasattr(out, "video") or hasattr(out, "frames") or (hasattr(out, "get") and ("videos" in out or "video" in out or "frames" in out)):
        # Video pipeline output
        logger.info("Video output detected")
        
        if hasattr(out, "videos") and out.videos is not None:
            videos = out.videos[0] if isinstance(out.videos, list) else out.videos
        elif hasattr(out, "video") and out.video is not None:
            videos = out.video[0] if isinstance(out.video, list) else out.video
        elif hasattr(out, "frames") and out.frames is not None:
            videos = out.frames[0] if isinstance(out.frames, list) else out.frames
        elif hasattr(out, "get"):
            videos_raw = out.get("videos")
            if videos_raw is None:
                videos_raw = out.get("video")
            if videos_raw is None:
                videos_raw = out.get("frames")
            videos = videos_raw[0] if isinstance(videos_raw, list) else videos_raw
            
        audio_raw = getattr(out, "audio", None)
        if audio_raw is None and hasattr(out, "get"):
            audio_raw = out.get("audio")
        audio = audio_raw[0] if isinstance(audio_raw, list) else audio_raw
        
        sampling_rate = getattr(out, "sampling_rate", None)
        if sampling_rate is None and hasattr(out, "get"):
            sampling_rate = out.get("sampling_rate", 24000)
        if sampling_rate is None:
            sampling_rate = 24000

        gen_time = time.time() - t_gen
        logger.info("Generation took %.1fs", gen_time)
        
        from diffusers.utils import encode_video
        output_path = Path(args.output)
        if output_path.suffix.lower() not in [".mp4", ".gif", ".avi", ".mov"]:
            output_path = output_path.with_suffix(".mp4")
            
        if isinstance(videos, torch.Tensor):
            # The MiniMax-H3 VAE output is ImageNet normalized.
            # We must denormalize it to [0, 1] before saving to avoid massive clipping
            # that causes visible 16x16 grid patches in the output.
            mean = torch.tensor([0.485, 0.456, 0.406], device=videos.device, dtype=videos.dtype).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=videos.device, dtype=videos.dtype).view(-1, 1, 1)
            
            # Identify the channel dimension to broadcast correctly
            if videos.shape[0] == 3 and len(videos.shape) == 4:
                # Shape is [C, T, H, W]
                mean = mean.unsqueeze(1) # [C, 1, 1, 1]
                std = std.unsqueeze(1)
            elif len(videos.shape) == 4 and videos.shape[1] == 3:
                # Shape is [T, C, H, W]
                mean = mean.unsqueeze(0) # [1, C, 1, 1]
                std = std.unsqueeze(0)
            
            # Apply denormalization
            videos = (videos * std) + mean
            videos = videos.clamp(0, 1)
            
        try:
            encode_video(
                videos,
                fps=24,  # MiniMax-H3 default FPS
                output_path=str(output_path),
                audio=audio,
                audio_sample_rate=sampling_rate,
            )
            logger.info("Saved to: %s", output_path.resolve())
        except Exception as e:
            logger.warning(f"encode_video failed, attempting manual video save: {e}")
            import torchvision.io
            if isinstance(videos, torch.Tensor):
                # Ensure it's [T, C, H, W]
                if videos.shape[0] == 3 and len(videos.shape) == 4:
                    videos = videos.permute(1, 0, 2, 3) # [C, T, H, W] -> [T, C, H, W]
                
                # Convert to [0, 255] uint8
                if videos.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                    videos = (videos * 255).clamp(0, 255).to(torch.uint8)
                torchvision.io.write_video(str(output_path), videos.permute(0, 2, 3, 1), fps=24, audio_array=audio, audio_fps=sampling_rate, audio_codec='aac')
                logger.info("Saved manually using torchvision.io.write_video to: %s", output_path.resolve())
        
    elif hasattr(out, "images"):
        # Image generation (Flux, SD3, etc)
        image = out.images[0]
        gen_time = time.time() - t_gen
        logger.info("Generation took %.1fs", gen_time)
        
        # ── Save ─────────────────────────────────────────────────────────────────
        output_path = Path(args.output)
        image.save(str(output_path))
        logger.info("Saved to: %s", output_path.resolve())
    else:
        # Fallback
        result = out[0][0] if hasattr(out, "__getitem__") else out
        gen_time = time.time() - t_gen
        logger.info("Generation took %.1fs", gen_time)
        
        if hasattr(result, "save"):
            output_path = Path(args.output)
            result.save(str(output_path))
            logger.info("Saved to: %s", output_path.resolve())

    # ── Budget report ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        limit_str = f"{args.vram_budget:.1f} GB" if args.vram_budget else "Dynamic"
        
        if args.vram_budget and peak_gb <= args.vram_budget:
            logger.info("\nPeak VRAM: %.3f GB / %s  [OK]\n", peak_gb, limit_str)
        elif args.vram_budget:
            logger.info("\nPeak VRAM: %.3f GB / %s  [EXCEEDED]\n", peak_gb, limit_str)
            print(f"WARNING: VRAM exceeded the {args.vram_budget:.1f} GB budget!", file=sys.stderr)
            return 2
        else:
            logger.info("\nPeak VRAM: %.3f GB (Dynamic Limit)  [OK]\n", peak_gb)

    return 0


if __name__ == "__main__":
    sys.exit(main())
