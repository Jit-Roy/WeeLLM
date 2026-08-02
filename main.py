"""
main.py -- WeeLLM: Layer-streaming inference for large diffusion models.

Fits multi-billion-parameter diffusion models under 4 GB VRAM and 8 GB RAM
without quantization or model reduction.

Usage
-----
    python main.py --model flux2-klein --prompt "A sunset over mountains"
    python main.py --model flux2-klein --height 768 --width 768 --steps 4 --seed 42
    python main.py --help
"""

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Default model directory names (auto-detected when --model_dir is not given)
# ---------------------------------------------------------------------------
_MODEL_DEFAULT_DIRS: dict = {
    "flux2-klein": "flux2-klein-4b",
    "flux2_klein": "flux2-klein-4b",
    # "sd35":       "sd3.5-medium",   # ← add when supported
    # "sdxl":       "sdxl-1.0",
}


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    from weellm import list_models  # import here so --help works even without torch

    available = ", ".join(list_models()) or "(none registered)"

    parser = argparse.ArgumentParser(
        prog="weellm",
        description=(
            "WeeLLM — Layer-streaming diffusion inference.\n"
            "Runs large models under 4 GB VRAM without quantization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
supported models:
  {available}

examples:
  python main.py --model flux2-klein --prompt "A majestic lion at golden hour"
  python main.py --model flux2-klein --height 768 --width 768 --steps 4
  python main.py --model flux2-klein --no_prefetch          # lower RAM usage
  python main.py --model flux2-klein --force_resplit        # rebuild shards
        """,
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="flux2-klein",
        metavar="NAME",
        help=f"Model to use. Available: {available}  (default: flux2-klein)",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        metavar="PATH",
        help="Override the model directory (auto-detected from --model if omitted)",
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
        help="Classifier-free guidance scale, 1.0 = disabled (default: 1.0)",
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
        help="Disable background prefetching (slower but ~400 MB less RAM)",
    )
    parser.add_argument(
        "--force_resplit", action="store_true",
        help="Force rebuilding of per-layer shards even if they already exist",
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

    # ── Resolve model directory ──────────────────────────────────────────────
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    elif args.model in _MODEL_DEFAULT_DIRS:
        model_dir = SCRIPT_DIR / _MODEL_DEFAULT_DIRS[args.model]
    else:
        model_dir = SCRIPT_DIR / args.model

    # ── Header ──────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  WeeLLM — Layer-Streaming Diffusion Inference")
    print(sep)
    print(f"  Model:    {args.model}  [{model_dir}]")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Size:     {args.width} x {args.height} px")
    print(f"  Steps:    {args.steps}  |  Guidance: {args.guidance_scale}  |  Seed: {args.seed}")
    print(f"  Dtype:    {args.dtype}  |  Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")
    print()

    # ── Load pipeline ────────────────────────────────────────────────────────
    from weellm import get_pipeline

    try:
        PipelineClass = get_pipeline(args.model)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    t_load = time.time()
    pipe = PipelineClass.from_pretrained(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        prefetch=not args.no_prefetch,
        force_resplit=args.force_resplit,
    )
    print(f"Pipeline loaded in {time.time() - t_load:.1f}s\n")

    # ── Generate ─────────────────────────────────────────────────────────────
    t_gen = time.time()
    image = pipe.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
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
