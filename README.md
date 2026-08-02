# WeeLLM

**Layer-streaming inference for large diffusion models — 4 GB VRAM, 8 GB RAM, no quantization.**

WeeLLM streams one transformer layer at a time from disk to GPU, inspired by [AirLLM](https://github.com/lyogavin/airllm) for language models. The full model weights never reside in VRAM simultaneously — only the currently-executing layer is loaded.

---

## Supported Models

| Model | Size | Min VRAM | Min RAM | Time (512×512, 4 steps) |
|---|---|---|---|---|
| `flux2-klein` | 4B params | **2 GB** | **2.3 GB** | ~61s on RTX 3050 |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python main.py --model flux2-klein --prompt "A majestic lion at golden hour"

# More options
python main.py --model flux2-klein \
    --prompt "A cyberpunk city at night" \
    --height 768 --width 768 \
    --steps 4 --seed 123 \
    --output my_image.png
```

---

## Project Structure

```
WeeLLM/
├── main.py                          # Model-agnostic CLI
├── requirements.txt
│
└── weellm/
    ├── __init__.py                  # Public API
    ├── registry.py                  # Model name → pipeline class map
    │
    ├── core/                        # Shared infrastructure
    │   ├── base_pipeline.py         # Abstract BasePipeline
    │   ├── base_streamer.py         # Abstract BaseStreamer
    │   └── utils.py                 # Memory utilities
    │
    └── models/
        └── flux2_klein/             # FLUX.2 Klein 4B
            ├── pipeline.py
            ├── transformer_streamer.py
            ├── text_encoder_streamer.py
            └── splitter.py
```

---

## Adding a New Diffusion Model

**3 steps:**

### 1. Create a model subpackage

```
weellm/models/your_model/
    __init__.py
    pipeline.py          # class LightYourModelPipeline(BasePipeline)
    unet_streamer.py     # (inherits BaseStreamer)
```

### 2. Implement the interface

```python
# weellm/models/your_model/pipeline.py
from weellm.core.base_pipeline import BasePipeline

class LightYourModelPipeline(BasePipeline):

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda", dtype=torch.bfloat16, **kwargs):
        # load VAE, scheduler, set up streaming for UNet / transformer
        ...

    def generate(self, prompt, height=512, width=512, num_inference_steps=20, **kwargs):
        # encode prompt -> denoise -> decode
        ...
        return image  # PIL.Image.Image
```

### 3. Register it

```python
# weellm/registry.py  (one line)
from .models.your_model import LightYourModelPipeline
register_model("your-model", LightYourModelPipeline)
```

**Done.** Now run:
```bash
python main.py --model your-model --prompt "..."
```

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GPU VRAM (4 GB budget)                                 │
│  ┌──────────────┐ ┌──────────────┐                      │
│  │  VAE ~174MB  │ │ Resident     │  ← always loaded     │
│  │  (resident)  │ │ modules ~390MB│                      │
│  └──────────────┘ └──────────────┘                      │
│  ┌──────────────────────────────┐                        │
│  │  Current layer  ~490MB       │  ← streamed in/out    │
│  │  (loaded from disk, evicted  │                        │
│  │   after forward pass)        │                        │
│  └──────────────────────────────┘                        │
│  Peak: ~2.0 GB  ✓  (50% headroom)                       │
└─────────────────────────────────────────────────────────┘
```

**Background pipeline:** while layer N runs on GPU, layer N+1 is already being loaded from disk directly to GPU via a background thread — eliminating I/O wait time.

---

## CLI Reference

```
usage: weellm [-h] [--model NAME] [--model_dir PATH] [--prompt TEXT]
                [--height H] [--width W] [--steps N] [--guidance_scale F]
                [--seed N] [--output PATH] [--dtype {bfloat16,float16,float32}]
                [--no_prefetch] [--force_resplit] [--vram_budget F]

options:
  --model NAME          Model identifier (default: flux2-klein)
  --model_dir PATH      Override model directory
  --prompt TEXT         Image generation prompt
  --height H            Output height in pixels (default: 512)
  --width W             Output width in pixels  (default: 512)
  --steps N             Denoising steps (default: 4)
  --guidance_scale F    CFG scale, 1.0=disabled (default: 1.0)
  --seed N              Random seed (default: 42)
  --output PATH         Output file (default: output.png)
  --dtype DTYPE         Compute dtype: bfloat16 | float16 | float32
  --no_prefetch         Disable background prefetching (saves ~400MB RAM)
  --force_resplit       Rebuild per-layer shard files
  --vram_budget F       VRAM limit for budget report (default: 4.0)
```
