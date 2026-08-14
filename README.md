# WeeLLM

**Layer-streaming inference for large diffusion models — Under 4 GB VRAM, no quantization**

WeeLLM streams one transformer layer at a time from disk to GPU for large models. The full model weights never reside in VRAM simultaneously — only the currently-executing layer is loaded.


---

## Supported Models

| Model                            | Parameters |   Peak VRAM |    Peak RAM |   Time (512×512) |
| -------------------------------- | ---------: | ----------: | ----------: | ---------------: |
| `FLUX.2-klein-4B`                |         4B |  **2.0 GB** |  **2.3 GB** |   ~61s · 4 steps |
| `Lumina-Image-2.0`               |        ~2B | **1.38 GB** |           — |                — |
| `FLUX.1-dev`                     |       ~12B | **1.51 GB** | **~2.0 GB** |                — |
| `FLUX.1-Kontext-dev`             |       ~12B | **1.51 GB** | **~2.0 GB** |                — |
| `FLUX.1-schnell`                 |       ~12B | **1.64 GB** | **1.68 GB** |  ~159s · 4 steps |
| `CogView4-6B`                    |       ~15B | **2.44 GB** | **1.92 GB** | ~411s · 10 steps |
| `SD 3.5 Medium`                  |        ~8B | **3.48 GB** | **3.96 GB** | ~219s · 20 steps |
| `Z-Image-Turbo`                  |       ~10B | **1.60 GB** | **1.70 GB** |  ~167s · 4 steps |
| `HiDream-I1-Full`                |       ~15B | **3.68 GB** | **2.02 GB** | ~797s · 10 steps |
| `SDXL` (Juggernaut XL v9)        |      ~6.6B | **2.98 GB** | **1.50 GB** | ~120s · 20 steps |
| `SD 1.5`                         |      ~1.7B | **0.90 GB** | **1.40 GB** |  ~68s · 20 steps |
| `Qwen-Image` (`Qwen/Qwen-Image`) |       ~20B | **2.47 GB** | **1.60 GB** | ~795s · 10 steps |
| `AuraFlow` (`fal/AuraFlow`)      |        ~4B | **1.34 GB** | **1.61 GB** | ~405s · 10 steps |
| `ERNIE-Image` (`Baidu/ERNIE-Image`)|    ~10B | **1.69 GB** | **2.54 GB** | ~123s · 5 steps (1024x1024) |

> The models run with **no quantization** on RTX-3050 — full bfloat16(1024 x 1024) weights streamed layer-by-layer from disk.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run directly from a Hugging Face repository ID!
python main.py --model black-forest-labs/FLUX.1-dev --prompt "A majestic lion at golden hour"

# Run from a local folder (skips download)
python main.py --model ./my-local-flux-model --prompt "A cyberpunk city at night"
```

---

## ⚠️ Cloud vs. Local Execution Warning

> [!WARNING]
> **This streaming architecture is specifically optimized for local execution on modern PCs equipped with NVMe SSDs and modern GPUs. You will experience slowdowns on standard cloud instances due to restricted disk I/O speeds.**

---

## 🛠 Troubleshooting: VRAM Spikes (e.g. 6GB+ on Kaggle T4)

If your Peak VRAM unexpectedly shoots up during generation (e.g., reaching 6 GB instead of the expected 2 GB), it is almost certainly a **hardware dtype compatibility issue**. 

When running on older GPUs like the NVIDIA T4 (Turing architecture) commonly found on Kaggle/Colab, the hardware **does not natively support `bfloat16`**. 
If you force `--dtype bfloat16`, PyTorch will silently disable memory-efficient FlashAttention kernels and fall back to the unoptimized **Math backend** for `scaled_dot_product_attention`. 

The Math backend materializes the full $N \times N$ attention matrix in VRAM, which for modern high-resolution diffusion models (like FLUX) causes a colossal memory allocation.

---

## Project Structure

```text
WeeLLM/
├── main.py                          # Universal CLI (auto-detects model type)
├── requirements.txt
│
└── weellm/
    ├── __init__.py                  # Public API
    ├── pipeline.py                  # Universal WeePipeline (loads any model via model_index.json)
    ├── seeker.py                    # Base Safetensors LiveSeeker
    ├── disk_seek.py                 # Disk-based Safetensors stream reader
    ├── ram_seek.py                  # RAM-cached Safetensors stream reader
    ├── utils.py                     # Memory & HF utilities
    │
    └── models/                      # Modular hook-based streamers
        ├── text_encoders/           # e.g., qwen3_for_causal_lm.py
        ├── transformers/            # e.g., flux_transformer_2d_model.py
        ├── unets/                   # Legacy models
        └── vaes/                    # e.g., lazy_vae.py
```

---

## Python API

```python
from weellm import WeePipeline

# Auto-detects model type from HF repo or local directory
pipe = WeePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", device="cuda", cache_to_ram=False)
image = pipe.generate(
    prompt="A serene Japanese zen garden at sunrise",
    height=512,
    width=512,
    num_inference_steps=4,
    seed=42,
)
image.save("output.png")
```