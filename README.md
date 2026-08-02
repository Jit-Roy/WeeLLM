# WeeLLM

**Layer-streaming inference for large diffusion models — Under 4 GB VRAM, no quantization**

WeeLLM streams one transformer layer at a time from disk to GPU for large models. The full model weights never reside in VRAM simultaneously — only the currently-executing layer is loaded.

WeeLLM uses a **Live Seek Architecture**: it reads weights directly out of Hugging Face safetensors files on the fly. This means **zero bytes of duplicated files** on your hard drive, and zero startup delay.

---

## Supported Models

| Model | Size | Peak VRAM | Peak RAM | Time (512×512, steps) |
|---|---|---|---|---|
| `flux2-klein` | 4B params | **2.0 GB** | **2.3 GB** | ~61s · 4 steps · RTX 3050 |
| `z-image-turbo` | ~10B params | **1.6 GB** | **1.7 GB** | ~167s · 4 steps · RTX 3050 |
| `sdxl` (Juggernaut XL v9) | ~6.6B params | **2.98 GB** | **1.5 GB** | ~120s · 20 steps · RTX 3050 |
| `sd15` (SD v1.5) | ~1.7B params | **0.90 GB** | **1.4 GB** | ~68s · 20 steps · RTX 3050 |

> The models run with **no quantization** — full bfloat16 weights streamed layer-by-layer from disk.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly from a Hugging Face repository ID!
# (It will automatically download to your HF cache if you don't have it)

# Flux2-Klein (4B)
python main.py --model black-forest-labs/FLUX.1-dev --prompt "A majestic lion at golden hour"

# Z-Image-Turbo (~10B) — photorealistic powerhouse
python main.py --model Tongyi-MAI/Z-Image-Turbo --prompt "A serene Japanese zen garden at sunrise, photorealistic, 8k"

# SDXL (Juggernaut XL v9 — 6.6B) — classic photorealistic quality
python main.py --model RunDiffusion/Juggernaut-XL-v9 \
  --prompt "A cyberpunk city at night with neon signs" \
  --steps 20 --guidance_scale 7.0

# Stable Diffusion 1.5 — ultra-lightweight, under 1 GB VRAM!
python main.py --model runwayml/stable-diffusion-v1-5 \
  --prompt "A cyberpunk city at night with neon signs" \
  --steps 20 --guidance_scale 7.5

# Run from a local folder (skips download)
python main.py --model ./my-local-flux-model --prompt "A cyberpunk city at night"
```

---

## ⚠️ Cloud vs. Local Execution Warning

> [!WARNING]
> **Do NOT try to run this natively on standard Kaggle or Google Colab environments!**

This streaming architecture is specifically optimized for local execution on modern PCs equipped with NVMe SSDs and modern GPUs. You will experience massive slowdowns on standard cloud instances.

---

## Model Setup

You can pass a **Hugging Face Repository ID** (e.g. `Tongyi-MAI/Z-Image-Turbo`) or an absolute/relative path to a local folder. 

If you pass a Hugging Face repo ID, WeeLLM will automatically download the necessary safetensors and config files using `huggingface_hub.snapshot_download` and cache them in your default Hugging Face cache folder.

---

## Project Structure

```
WeeLLM/
├── main.py                          # Model-agnostic CLI
├── requirements.txt
│
└── weellm/
    ├── __init__.py                  # Public API
    ├── auto.py                      # Auto-router (detects model from config & downloads)
    ├── registry.py                  # Model name → pipeline class map
    │
    ├── core/                        # Shared infrastructure
    │   ├── encoders/                # Shared encoders
    │   │   ├── qwen3_streamer.py    # Qwen3 text encoder (Flux / Z-Image)
    │   │   └── clip_streamer.py     # CLIP text encoder streamer (SD1.5 / SDXL)
    │   ├── base_pipeline.py         # Abstract BasePipeline
    │   ├── base_streamer.py         # Abstract BaseStreamer
    │   ├── live_seek.py             # SafetensorsLiveSeeker (zero-duplication disk reader)
    │   └── utils.py                 # Memory & HF resolution utilities
    │
    └── models/
        ├── flux2_klein/             # FLUX.2 Klein 4B
        │   ├── pipeline.py
        │   └── transformer_streamer.py
        │
        ├── z_image_turbo/           # Z-Image-Turbo ~10B
        │   ├── pipeline.py
        │   └── transformer_streamer.py
        │
        ├── sdxl/                   # Stable Diffusion XL (Juggernaut XL v9, ~6.6B)
        │   ├── pipeline.py
        │   └── unet_streamer.py      # Shared with SD1.5
        │
        └── sd15/                   # Stable Diffusion 1.5 (~1.7B)
            └── pipeline.py          # Reuses unet_streamer + clip_streamer
```

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GPU VRAM (4 GB budget)                                 │
│  ┌──────────────┐ ┌──────────────┐                      │
│  │  VAE ~160MB  │ │ Resident     │  ← always loaded     │
│  │  (resident)  │ │ modules ~50MB│                      │
│  └──────────────┘ └──────────────┘                      │
│  ┌──────────────────────────────┐                       │
│  │  Current layer  ~800MB       │  ← streamed in/out    │
│  │  (read direct from HF file)  │                       │
│  └──────────────────────────────┘                       │
│  Peak: ~1.6 GB  ✓  (60% headroom)                       │
└─────────────────────────────────────────────────────────┘
```

**Background pipeline:** while layer N runs on GPU, layer N+1 is already being loaded from disk directly to GPU via a background thread — eliminating I/O wait time.

---

## Python API

```python
from weellm.auto import WeePipeline

# Auto-detects model type from HF repo or local directory
pipe = WeePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", device="cuda")
image = pipe.generate(
    prompt="A serene Japanese zen garden at sunrise",
    height=512,
    width=512,
    num_inference_steps=4,
    seed=42,
)
image.save("output.png")

# SDXL (Juggernaut XL v9)
pipe = WeePipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", device="cuda")
image = pipe.generate(
    prompt="A photorealistic cyberpunk city at night with neon signs",
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.0,
    seed=42,
)
image.save("sdxl_output.png")

# Stable Diffusion 1.5 — under 1 GB VRAM!
pipe = WeePipeline.from_pretrained("runwayml/stable-diffusion-v1-5", device="cuda")
image = pipe.generate(
    prompt="A photorealistic cyberpunk city at night with neon signs",
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=42,
)
image.save("sd15_output.png")
```

---

## CLI Reference

```
usage: weellm [-h] --model ID_OR_PATH [--prompt TEXT]
                [--height H] [--width W] [--steps N] [--guidance_scale F]
                [--seed N] [--output PATH] [--dtype {bfloat16,float16,float32}]
                [--no_prefetch] [--vram_budget F]

options:
  --model ID_OR_PATH    Hugging Face repo ID or path to local model directory
  --prompt TEXT         Image generation prompt
  --height H            Output height in pixels (default: 512)
  --width W             Output width in pixels  (default: 512)
  --steps N             Denoising steps (default: 4)
  --guidance_scale F    CFG scale (default: 0.0 for z-image-turbo)
  --seed N              Random seed (default: 42)
  --output PATH         Output file (default: output.png)
  --dtype DTYPE         Compute dtype: bfloat16 | float16 | float32
  --no_prefetch         Disable background prefetching (saves ~400MB RAM)
  --vram_budget F       VRAM limit for budget report (default: 4.0)
```
