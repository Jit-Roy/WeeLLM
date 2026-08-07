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

import argparse
import sys
import time
from pathlib import Path

import torch


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
    parser.add_argument("--height", type=int, default=512, help="Output height in pixels (default: 512)")
    parser.add_argument("--width",  type=int, default=512, help="Output width in pixels  (default: 512)")
    parser.add_argument("--steps",  type=int, default=4,   help="Denoising steps         (default: 4)")
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="Classifier-free guidance scale (default: 1.0, 1.0 = disabled)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

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
        "--vram_budget", type=float, default=4.0,
        help="VRAM budget in GB for the pass/fail report (default: 4.0)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Device & dtype ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # ── Header ──────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  WeeLLM — Layer-Streaming Diffusion Inference")
    print(sep)
    print(f"  Model:    {args.model}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Size:     {args.width} x {args.height} px")
    print(f"  Steps:    {args.steps}  |  Guidance: {args.guidance_scale}  |  Seed: {args.seed}")
    print(f"  Dtype:    {args.dtype}  |  Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")
    print()

    # ── Load pipeline ────────────────────────────────────────────────────────
    from weellm import WeePipeline

    t_load = time.time()
    try:
        pipe = WeePipeline.from_pretrained(
            model_dir=args.model,
            device=device,
            torch_dtype=dtype,
            prefetch=not args.no_prefetch,
            cache_to_ram=args.cache_to_ram,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
        
    print(f"Pipeline loaded in {time.time() - t_load:.1f}s\n")

    # ── Generate ─────────────────────────────────────────────────────────────
    t_gen = time.time()
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed) if args.seed is not None else None
    out = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    image = out.images[0] if hasattr(out, "images") else out[0][0]
    gen_time = time.time() - t_gen
    print(f"Generation took {gen_time:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    image.save(str(output_path))
    print(f"Saved to: {output_path.resolve()}")

    # ── Budget report ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        peak  = torch.cuda.max_memory_allocated() / 1e9
        limit = args.vram_budget
        ok    = peak <= limit
        print(f"\nPeak VRAM: {peak:.3f} GB / {limit:.1f} GB  [{'OK' if ok else 'EXCEEDED'}]")
        if not ok:
            print(f"WARNING: VRAM exceeded the {limit:.1f} GB budget!", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
