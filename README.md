![WeeLLM Banner](docs/banner.png)

<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  <br>
  <a href="https://www.kaggle.com/code/freedomfighter1290/weellm"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>

# Layer-streaming inference for large diffusion models — Under 4 GB VRAM, no quantization

WeeLLM dynamically streams transformer layers to the GPU for massive models. Instead of forcing the entire model into VRAM, it intelligently pins as many blocks as your hardware allows, and seamlessly streams the rest layer-by-layer in the background — enabling massive models to run smoothly on budgets as low as 4GB VRAM.

---

## Some Benchmarks(For RTX 3050 Having <4GB Vram)

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
| `Krea-2-Turbo`                     |       ~13B |   **3  GB** | **3.23 GB** | ~810s · 10 steps |
| `MiniMax-H3`                       |       ~34B | **3.14 GB** |           — |                — |

> The models run with **no quantization** on RTX-3050 — full bfloat16 (1024x1024) weights streamed layer-by-layer to GPU.

<div align="center">
  <img src="docs/bar_chart.png" alt="Performance Bar Chart" width="85%">
</div>

# Supported Model List

## Image Generation Models

- SDXL
- SD 1.5
- SD 2.0
- SD 2.1
- SD 3 Medium
- SD 3.5 Medium
- SD 3.5 Large
- Flux.1 Dev
- Flux.1 Schnell
- Flux.1 Kontext
- Flux.1 Krea
- Flux.2 Klein 4B
- Flux.2 Klein 9B
- CogView 4
- LongCat Image
- Z-Image Turbo
- Z-Image Base
- Krea 2 Turbo
- Krea 2 Raw
- Qwen Image
- Ideogram 4
- ERNIE-Image
- ERNIE-Image-Turbo
- Lumina Image 2.0
- Hidream I1 Full
- Auraflow

## Image Edit Models

- SDXL Img2Img
- SD 1.5 Img2Img
- SD 2.1 Img2Img
- SD 3 Medium Img2Img
- SD 3.5 Medium Img2Img
- SD 3.5 Large Img2Img
- Flux.1 Kontext 
- Flux.1 Fill Dev
- Flux.2 Klein 4B
- Flux.2 Klein 9B
- LongCat Image Edit
- Qwen Image Edit
- HiDream E1 Full

## Video Generation Models

- MiniMax-H3 (FL2VA)
- LTX 2.5 (Text To Video + Image To Video)

---

## TODOs

- [ ] **ControlNet / T2I-Adapter Integration:** Enable structural conditioning (canny, depth, pose) while maintaining strict VRAM streaming budgets.
- [ ] **LoRA Support:** Dynamically load and apply LoRA weights during the layer-streaming process without bloating system Vram and RAM.

---

## Quick Start

### Local Setup
```bash
git clone https://github.com/Jit-Roy/weellm.git
cd weellm
pip install -r requirements.txt
```

### Kaggle / Colab Setup
```bash
!git clone https://github.com/Jit-Roy/weellm.git
%cd weellm
!pip install -r requirements.txt
```

### Kaggle / Colab Setup (via pip, no git clone needed)
```bash
!pip install -q git+https://github.com/Jit-Roy/weellm
```

### Running the CLI
Run the `main.py` script and explicitly pass your VRAM and RAM budgets (in GB) to ensure you stay within your hardware limits.

```bash
# Run directly from a Hugging Face repository ID!
python main.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A majestic lion at golden hour" \
    --negative_prompt "blurry, distorted, low quality" \
    --height 1024 \
    --width 1024 \
    --steps 20 \
    --guidance_scale 3.5 \
    --seed 42 \
    --dtype bfloat16 \
    --vram_budget 4 \
    --ram_budget 4 \
    --output "flux_lion.png" \
    --verbose

# Run Text-to-Video from a Hugging Face repository ID!
python main.py \
    --model "Lightricks/LTX-Video" \
    --prompt "A drone flying over a snowy mountain peak at sunrise." \
    --negative_prompt "worst quality, inconsistent motion, blurry" \
    --height 512 \
    --width 704 \
    --steps 40 \
    --guidance_scale 3.0 \
    --seed 12345 \
    --dtype bfloat16 \
    --vram_budget 4 \
    --ram_budget 4 \
    --output "output_video.mp4" \
    --verbose
```

---

## ⚠️ Cloud vs. Local Execution Warning

> [!WARNING]
> **This streaming architecture is specifically optimized for local execution on modern PCs equipped with NVMe SSDs and modern GPUs. You will experience slowdowns on standard cloud instances due to restricted disk I/O speeds.**

---

## ⚠️ Autoregressive & Vision-Language Model Limitations

> Dense autoregressive text-to-image models (like GLM-Image or other Vision-Language Models) are **not practically supported** by SSD streaming without quantization.
> Unlike diffusion models which process latents in a few discrete steps (e.g., 4 to 20 steps), autoregressive models must generate tokens sequentially. A 1024x1024 image requires generating over 1000 tokens. Generating 1000 tokens means doing 1000 full forward passes through the massive text encoder. 
> Streaming a 20GB model from an SSD 1000 times requires transferring ~20 Terabytes of data, which mathematically takes **several hours** to generate a single image. While the code technically executes without crashing under 4GB VRAM, the wait time is completely impractical. If you wish to run these autoregressive models efficiently, you must use 4-bit/8-bit quantization so the weights can sit entirely in System RAM or VRAM.

---

## Python API

### Text to Image

```python
from weellm import WeePipeline
import torch

pipe = WeePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    device="cuda", 
    torch_dtype=torch.bfloat16,
    vram_budget=4, 
    ram_budget=4
)

image = pipe.generate(
    prompt="A serene Japanese zen garden at sunrise",
    negative_prompt="blurry, low quality",
    height=512,
    width=512,
    num_inference_steps=4,
    guidance_scale=1.5,
    seed=42,
)
image.save("output.png")
```

### Text + Image to Image (Image-to-Image)

```python
from weellm import WeePipeline
from PIL import Image
import torch

pipe = WeePipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    device="cuda", 
    torch_dtype=torch.bfloat16,
    vram_budget=4, 
    ram_budget=4
)

init_image = Image.open("input.jpg").convert("RGB")
image = pipe.generate(
    prompt="A futuristic cyberpunk city",
    negative_prompt="blurry, distorted",
    image=init_image,
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=3.5,
    seed=42,
)
image.save("output_i2i.png")
```

### Text to Video

```python
from weellm import WeePipeline
from weellm.utils import export_to_video
import torch

pipe = WeePipeline.from_pretrained(
    "Lightricks/LTX-Video", 
    device="cuda", 
    torch_dtype=torch.bfloat16,
    vram_budget=4, 
    ram_budget=4
)

video_frames = pipe.generate(
    prompt="A drone flying over a snowy mountain peak at sunrise.",
    negative_prompt="worst quality, inconsistent motion",
    height=512,
    width=704,
    num_inference_steps=40,
    guidance_scale=3.0,
    seed=42,
)
export_to_video(video_frames, "output_video.mp4", fps=24)
```

### Text + Image to Video

```python
from weellm import WeePipeline
from weellm.utils import export_to_video
from PIL import Image
import torch

pipe = WeePipeline.from_pretrained(
    "Lightricks/LTX-Video", 
    device="cuda", 
    torch_dtype=torch.bfloat16,
    vram_budget=4, 
    ram_budget=4
)

start_image = Image.open("start_frame.jpg").convert("RGB")
video_frames = pipe.generate(
    prompt="The camera pans slowly across the room.",
    negative_prompt="worst quality, inconsistent motion",
    image=start_image,
    height=512,
    width=704,
    num_inference_steps=40,
    guidance_scale=3.0,
    seed=42,
)
export_to_video(video_frames, "output_i2v.mp4", fps=24)
```

### Using GGUF Models

You can use quantized `.gguf` weights for the transformer or text encoders to save disk space and RAM. The weights will be streamed and dequantized on the fly.

```python
from weellm import WeePipeline
import torch

pipe = WeePipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", 
    transformer_path="unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf",
    text_encoder_path="unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q5_K_M.gguf",
    # If using SD/SDXL models, use `unet_path` instead of `transformer_path`
    # unet_path="path/to/juggernaut-xl-v9-Q8_0.gguf",
    device="cuda", 
    torch_dtype=torch.bfloat16,
    vram_budget=4, 
    ram_budget=4,
    prefetch=False  # Recommended when using GGUF to avoid GPU contention
)

image = pipe.generate(
    prompt="A majestic lion at golden hour",
    height=1024,
    width=1024,
    num_inference_steps=4,
)
image.save("flux_klein_gguf.png")
```

---

## Star History

<a href="https://www.star-history.com/?repos=Jit-Roy%2FWeeLLM&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Jit-Roy/WeeLLM&type=date&theme=dark&legend=top-left&sealed_token=TuDBj5rYLt1aHcQBy3cpkcR05xvo_j5LhHv2wILO8AJdNy2Uj6pmXOX4a9Wmw2zoY2Py1flJ0nlMtc-fjp_CPPkCKNc1LdNiyy7zoUJarOS2xCbQbd48NkZpd8HA8hpepYkMxmiPV0yGtDsC_PmWiKiMTOtFeoHZB7-ryg9sVk8dC-Uam2O69kAFs47K" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Jit-Roy/WeeLLM&type=date&legend=top-left&sealed_token=TuDBj5rYLt1aHcQBy3cpkcR05xvo_j5LhHv2wILO8AJdNy2Uj6pmXOX4a9Wmw2zoY2Py1flJ0nlMtc-fjp_CPPkCKNc1LdNiyy7zoUJarOS2xCbQbd48NkZpd8HA8hpepYkMxmiPV0yGtDsC_PmWiKiMTOtFeoHZB7-ryg9sVk8dC-Uam2O69kAFs47K" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Jit-Roy/WeeLLM&type=date&legend=top-left&sealed_token=TuDBj5rYLt1aHcQBy3cpkcR05xvo_j5LhHv2wILO8AJdNy2Uj6pmXOX4a9Wmw2zoY2Py1flJ0nlMtc-fjp_CPPkCKNc1LdNiyy7zoUJarOS2xCbQbd48NkZpd8HA8hpepYkMxmiPV0yGtDsC_PmWiKiMTOtFeoHZB7-ryg9sVk8dC-Uam2O69kAFs47K" />
 </picture>
</a>
