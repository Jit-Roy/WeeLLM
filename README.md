![WeeLLM Banner](docs/banner.png)

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Diffusers-0.31+-blue.svg?logo=huggingface" alt="Diffusers">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</div>

**Layer-streaming inference for large diffusion models — Under 4 GB VRAM, no quantization**

WeeLLM streams one transformer layer at a time from disk to GPU for large models. The full model weights never reside in VRAM simultaneously — only the currently-executing layer is loaded.


---

## Some Benchmarks

| Model                              | Parameters |   Peak VRAM |    Peak RAM | Time (1024x1024) |
| --------------------------------   | ---------: | ----------: | ----------: | ---------------: |
| `SDXL` (Juggernaut XL v9)          |      ~6.6B | **2.98 GB** | **1.50 GB** | ~120s · 20 steps |
| `SD 1.5`                           |      ~1.7B | **0.90 GB** | **1.40 GB** |  ~68s · 20 steps |
| `SD 3.5 Medium`                    |        ~8B | **3.48 GB** | **3.96 GB** | ~219s · 20 steps |
| `FLUX.1-dev`                       |       ~12B | **1.51 GB** | **~2.0 GB** |                — |
| `FLUX.1-Kontext-dev`               |       ~12B | **1.51 GB** | **~2.0 GB** |                — |
| `FLUX.1-schnell`                   |       ~12B | **1.64 GB** | **1.68 GB** |  ~159s · 4 steps |
| `FLUX.2-klein-4B`                  |         4B |  **2.0 GB** |  **2.3 GB** |   ~61s · 4 steps |
| `Lumina-Image-2.0`                 |        ~2B | **1.38 GB** |           — |                — |
| `CogView4-6B`                      |       ~15B | **2.44 GB** | **1.92 GB** | ~411s · 10 steps |
| `Z-Image-Turbo`                    |       ~10B | **1.60 GB** | **1.70 GB** |  ~167s · 4 steps |
| `HiDream-I1-Full`                  |       ~15B | **3.68 GB** | **2.02 GB** | ~797s · 10 steps |
| `LongCat-Image`                    |        ~9B | **2.19 GB** |           — | ~200s · 10 steps |
| `Qwen-Image` (`Qwen/Qwen-Image`)   |       ~20B | **2.47 GB** | **1.60 GB** | ~795s · 10 steps |
| `AuraFlow` (`fal/AuraFlow`)        |        ~4B | **1.34 GB** | **1.61 GB** | ~405s · 10 steps |
| `ERNIE-Image` (`Baidu/ERNIE-Image`)|       ~10B | **1.69 GB** | **2.54 GB** | ~123s · 5 steps  |

<div align="center">
  <img src="docs/bar_chart.png" alt="Performance Bar Chart" width="85%">
</div>

# Supported Model List

- SDXL
- SD 1.5
- SD 2.0
- SD 3.5 Medium
- SD 3.5 Large
- Flux.1 Dev
- Flux.1 Schnell
- Flux.1 Kontext
- Flux.1 Krea
- Flux Fill
- Flux.2 Klein 4B
- Flux.2 Klein 9B
- CogView 4
- LongCat Image
- Z-Image Turbo
- Z-Image Base
- Krea 2 Turbo
- Krea 2 Raw
- Qwen Image
- Qwen Image Edit
- Ideogram 4
- ERNIE-Image
- ERNIE-Image-Turbo
- Lumina Image 2.0
- Hidream I1 Full
- Auraflow

> The models run with **no quantization** on RTX-3050 — full bfloat16 (1024x1024) weights streamed layer-by-layer from disk.

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

## ⚠️ Autoregressive & Vision-Language Model Limitations

> [!WARNING]
> Dense autoregressive text-to-image models (like GLM-Image or other Vision-Language Models) are **not practically supported** by SSD streaming without quantization.
> Unlike diffusion models which process latents in a few discrete steps (e.g., 4 to 20 steps), autoregressive models must generate tokens sequentially. A 1024x1024 image requires generating over 1000 tokens. Generating 1000 tokens means doing 1000 full forward passes through the massive text encoder. 
> Streaming a 20GB model from an SSD 1000 times requires transferring ~20 Terabytes of data, which mathematically takes **several hours** to generate a single image. While the code technically executes without crashing under 4GB VRAM, the wait time is completely impractical. If you wish to run these autoregressive models efficiently, you must use 4-bit/8-bit quantization so the weights can sit entirely in System RAM or VRAM.

---

## 🛠 Troubleshooting: VRAM Spikes (e.g. 6GB+ on Kaggle T4)

If your Peak VRAM unexpectedly shoots up during generation (e.g., reaching 6 GB instead of the expected 2 GB), it is almost certainly a **hardware dtype compatibility issue**. 

When running on older GPUs like the NVIDIA T4 (Turing architecture) commonly found on Kaggle/Colab, the hardware **does not natively support `bfloat16`**. 
If you force `--dtype bfloat16`, PyTorch will silently disable memory-efficient FlashAttention kernels and fall back to the unoptimized **Math backend** for `scaled_dot_product_attention`. 

The Math backend materializes the full $N \times N$ attention matrix in VRAM, which for modern high-resolution diffusion models (like FLUX) causes a colossal memory allocation.

---

## Python API

```python
from weellm import WeePipeline

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

---

## Resources

- **Kaggle Notebook**: [WeeLLM Implementation & Benchmarks](https://www.kaggle.com/code/freedomfighter1290/weellm) — Interactive demonstrations and performance analysis