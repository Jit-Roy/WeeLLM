# WeeLLM

**Layer-streaming inference for large diffusion models — Under 4 GB VRAM, no quantization**

WeeLLM streams one transformer layer at a time from disk to GPU for large models. The full model weights never reside in VRAM simultaneously — only the currently-executing layer is loaded.

WeeLLM uses a **Live Seek Architecture**: it reads weights directly out of Hugging Face safetensors files on the fly. This means **zero bytes of duplicated files** on your hard drive, and zero startup delay.

---

## Supported Models

| Model | Size | Peak VRAM | Peak RAM | Time (512×512, 4–8 steps) |
|---|---|---|---|---|
| `flux2-klein` | 4B params | **2.0 GB** | **2.3 GB** | ~61s on RTX 3050 |
| `z-image-turbo` | ~10B params | **1.6 GB** | **1.7 GB** | ~167s on RTX 3050 |

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
    │   ├── encoders/                # Shared encoders (e.g. Qwen3)
    │   │   └── qwen3_streamer.py
    │   ├── base_pipeline.py         # Abstract BasePipeline
    │   ├── base_streamer.py         # Abstract BaseStreamer
    │   ├── live_seek.py             # LiveSeeker (zero-duplication disk reader)
    │   └── utils.py                 # Memory & HF resolution utilities
    │
    └── models/
        ├── flux2_klein/             # FLUX.2 Klein 4B
        │   ├── pipeline.py
        │   └── transformer_streamer.py
        │
        └── z_image_turbo/           # Z-Image-Turbo ~10B
            ├── pipeline.py
            └── transformer_streamer.py
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
