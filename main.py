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

    # Model selection
    parser.add_argument(
        "--model", type=str, required=True,
        metavar="ID_OR_PATH",
        help="Hugging Face repo ID or path to local model directory (e.g. Tongyi-MAI/Z-Image-Turbo)",
    )

    # Generation parameters
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
    parser.add_argument(
        "--guidance_scale", type=float, default=None,
        help="Classifier-free guidance scale (default: use pipeline default, usually 4.5 for SD3/LongCat or 1.0/3.5 for Flux)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (default: -1 for random)")
    
    # Image-to-Image / Editing
    parser.add_argument(
        "--image", type=str, default="",
        help="Path to an input image for image-to-image or editing tasks",
    )
    parser.add_argument(
        "--strength", type=float, default=0.8,
        help="Denoising strength for image-to-image (default: 0.8)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="output.png",
        help="Output image file path (default: output.png)",
    )

    # Dtype
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype (default: bfloat16)",
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
        "--vae_tile_size", type=int, default=256,
        help="VAE tiling minimum tile size in pixels (default: 256). Smaller = less VRAM spike, "
             "but possible tiling artifacts at boundaries.",
    )
    parser.add_argument(
        "--vram_budget", type=float, default=4.0,
        help="VRAM budget in GB for the pass/fail report (default: 4.0)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load the pipeline and verify setup without running inference. Useful for CI/testing.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug logging (including per-layer VRAM tracking).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.image:
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
        if args.width is None: args.width = 1024
        if args.height is None: args.height = 1024

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
    if args.image:
        from weellm import WeeImagePipeline as PipelineClass
        from PIL import Image
        logger.info("  Mode:     Image-to-Image / Edit (Input: %s)", args.image)
        try:
            input_image = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(f"ERROR: Could not load input image: {e}", file=sys.stderr)
            return 1
    else:
        from weellm import WeePipeline as PipelineClass
        input_image = None

    t_load = time.time()
    try:
        pipe = PipelineClass.from_pretrained(
            model_dir=args.model,
            device=device,
            torch_dtype=dtype,
            prefetch=not args.no_prefetch,
            cache_to_ram=args.cache_to_ram,
            vae_tile_size=args.vae_tile_size,
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
    if args.guidance_scale is not None:
        call_kwargs["guidance_scale"] = args.guidance_scale

    if input_image is not None:
        call_kwargs["image"] = input_image
        call_kwargs["strength"] = args.strength
        
    if args.negative_prompt:
        call_kwargs["negative_prompt"] = args.negative_prompt

    out   = pipe(**call_kwargs)
    image = out.images[0] if hasattr(out, "images") else out[0][0]
    gen_time = time.time() - t_gen
    logger.info("Generation took %.1fs", gen_time)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    image.save(str(output_path))
    logger.info("Saved to: %s", output_path.resolve())

    # ── Budget report ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        peak  = torch.cuda.max_memory_allocated() / 1e9
        limit = args.vram_budget
        ok    = peak <= limit
        logger.info("\nPeak VRAM: %.3f GB / %.1f GB  [%s]", peak, limit, "OK" if ok else "EXCEEDED")
        if not ok:
            print(f"WARNING: VRAM exceeded the {limit:.1f} GB budget!", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
