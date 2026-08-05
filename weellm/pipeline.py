import os
import json
import gc
import importlib
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from weellm.base_pipeline import BasePipeline
from weellm.utils import clean_memory, report_memory
import math

def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (math.log(max_seq_len) - math.log(base_seq_len))
    b = base_shift - m * math.log(base_seq_len)
    mu = m * math.log(image_seq_len) + b
    return mu


class WeePipeline(BasePipeline):
    def __init__(
        self,
        model_dir: Path,
        transformer,
        text_encoders: Dict[str, Any],
        tokenizers: Dict[str, Any],
        vae,
        scheduler,
        transformer_class_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        self.model_dir = model_dir
        self.transformer = transformer
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.vae = vae
        self.scheduler = scheduler
        self.transformer_class_name = transformer_class_name
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        **kwargs
    ):
        model_dir_str = str(model_dir)
        if not Path(model_dir_str).exists():
            print(f"Path '{model_dir_str}' not found locally. Attempting to download from Hugging Face Hub...")
            try:
                from huggingface_hub import snapshot_download
                
                print(f"Fetching model_index.json from '{model_dir_str}'...")
                index_dir = snapshot_download(model_dir_str, allow_patterns=["model_index.json"])
                index_path = Path(index_dir) / "model_index.json"
                
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                    
                allow_patterns = ["model_index.json"]
                for key in index_data:
                    if isinstance(index_data[key], list) and len(index_data[key]) == 2:
                        allow_patterns.append(f"{key}/*")
                
                print(f"Downloading only required components: {allow_patterns}")
                model_dir_str = snapshot_download(model_dir_str, allow_patterns=allow_patterns)
            except ImportError:
                raise ImportError("huggingface_hub is required to download models. Please install it with 'pip install huggingface_hub'.")
            except Exception as e:
                raise ValueError(f"Failed to download '{model_dir_str}' from Hugging Face Hub: {e}")

        model_dir = Path(model_dir_str)
        index_path = model_dir / "model_index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
            
        print("\n============================================================")
        print(f"  WeePipeline -- Loading {index.get('_class_name', 'Unknown')}")
        print("============================================================\n")
        
        # 1. Tokenizers & Scheduler
        print("[1/4] Loading Tokenizers and Scheduler ...")
        tokenizers = {}
        for key in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
            if key in index:
                from transformers import AutoTokenizer
                try:
                    tokenizers[key] = AutoTokenizer.from_pretrained(str(model_dir / key))
                except Exception as e:
                    print(f"Warning: Failed to load {key}: {e}")

        scheduler_cls = getattr(importlib.import_module("diffusers"), index["scheduler"][1])
        scheduler = scheduler_cls.from_pretrained(str(model_dir), subfolder="scheduler")
        
        # 2. VAE (Lazy Loaded on Meta Device)
        print("\n[2/4] Initializing VAE (Lazy loading on meta device) ...")
        from weellm.models.vaes.lazy_vae import LazyVAEStreamer
        vae_dir = model_dir / "vae"
        
        lazy_vae = LazyVAEStreamer.from_pretrained(
            vae_dir,
            device=device,
            dtype=dtype,
            cache_to_ram=cache_to_ram
        )
        vae = lazy_vae.model

        # 3. Text Encoders (Streamed or Resident)
        print("\n[3/4] Preparing Text Encoders ...")
        TE_MAP = {
            "CLIPTextModel": "weellm.models.text_encoders.clip_text_model",
            "CLIPTextModelWithProjection": "weellm.models.text_encoders.clip_text_model",
            "T5EncoderModel": "weellm.models.text_encoders.t5_encoder_model",
            "UMT5EncoderModel": "weellm.models.text_encoders.umt5_encoder_model",
            "Qwen2ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen3ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen3Model": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen2_5_VLForConditionalGeneration": "weellm.models.text_encoders.qwen2_5_vl_for_conditional_generation",
            "Mistral3ForConditionalGeneration": "weellm.models.text_encoders.mistral3_for_conditional_generation",
            "GlmModel": "weellm.models.text_encoders.glm_model",
            "Gemma2Model": "weellm.models.text_encoders.gemma2_model",
            "LlamaForCausalLM": "weellm.models.text_encoders.llama_for_causal_lm",
        }
        
        text_encoders = {}
        for key in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"]:
            if key in index:
                hf_cls_name = index[key][1]
                if hf_cls_name in TE_MAP:
                    module_path = TE_MAP[hf_cls_name]
                    streamer_cls_name = "CLIPTextModelStreamer" if "CLIP" in hf_cls_name else hf_cls_name + "Streamer"
                    module = importlib.import_module(module_path)
                    te_cls = getattr(module, streamer_cls_name)
                    
                    tok_key = key.replace("text_encoder", "tokenizer")
                    te_path = str(model_dir / key)
                        
                    if "Qwen" in hf_cls_name or "Mistral" in hf_cls_name or "Llama" in hf_cls_name:
                        text_encoders[key] = te_cls.from_pretrained(model_dir=te_path, tokenizer=tokenizers.get(tok_key), device=device, dtype=dtype, cache_to_ram=cache_to_ram)
                    elif "CLIP" in hf_cls_name:
                        hf_module = importlib.import_module("transformers")
                        hf_cls = getattr(hf_module, hf_cls_name)
                        text_encoders[key] = te_cls.from_pretrained(hf_cls, str(model_dir), key, device=device, dtype=dtype, output_hidden_states=True, cache_to_ram=cache_to_ram)
                    else:
                        text_encoders[key] = te_cls.from_pretrained(model_dir=te_path, device=device, dtype=dtype, cache_to_ram=cache_to_ram)
                else:
                    hf_module = importlib.import_module("transformers")
                    hf_cls = getattr(hf_module, hf_cls_name)
                    text_encoders[key] = hf_cls.from_pretrained(str(model_dir / key), torch_dtype=dtype).to(device)
                    text_encoders[key].eval()

        # 4. Transformer / UNet
        print("\n[4/4] Preparing Transformer / UNet ...")
        transformer_key = "transformer" if "transformer" in index else "unet"
        transformer_class_name = index[transformer_key][1]
        
        TR_MAP = {
            "FluxTransformer2DModel": "weellm.models.transformers.flux_transformer_2d_model",
            "Flux2Transformer2DModel": "weellm.models.transformers.flux2_transformer_2d_model",
            "ZImageTransformer2DModel": "weellm.models.transformers.z_image_transformer_2d_model",
            "SD3Transformer2DModel": "weellm.models.transformers.sd3_transformer_2d_model",
            "QwenImageTransformer2DModel": "weellm.models.transformers.qwen_image_transformer_2d_model",
            "CogView4Transformer2DModel": "weellm.models.transformers.cogview4_transformer_2d_model",
            "Lumina2Transformer2DModel": "weellm.models.transformers.lumina2_transformer_2d_model",
            "AuraFlowTransformer2DModel": "weellm.models.transformers.auraflow_transformer_2d_model",
            "HiDreamImageTransformer2DModel": "weellm.models.transformers.hidream_transformer_2d_model",
            "UNet2DConditionModel": "weellm.models.unets.unet_2d_condition_model"
        }
        
        module_path = TR_MAP.get(transformer_class_name, "")
        if not module_path:
            raise ValueError(f"Unsupported architecture: {transformer_class_name}")
            
        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, transformer_class_name + "Streamer")
        
        if transformer_key == "unet":
            transformer = transformer_cls.from_pretrained(str(model_dir), device, dtype, prefetch, cache_to_ram=cache_to_ram)
        else:
            transformer = transformer_cls.from_pretrained(model_dir / transformer_key, device=device, dtype=dtype, prefetch=prefetch, cache_to_ram=cache_to_ram)
        
        print("\n============================================================")
        print("  WeePipeline ready.")
        print("============================================================\n")
        
        return cls(
            model_dir=model_dir,
            transformer=transformer,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            vae=vae,
            scheduler=scheduler,
            transformer_class_name=transformer_class_name,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
    @property
    def _text_encoder(self): return self.text_encoders.get("text_encoder")
    @property
    def _text_encoder_1(self): return self.text_encoders.get("text_encoder")
    @property
    def _text_encoder_2(self): return self.text_encoders.get("text_encoder_2")
    @property
    def _text_encoder_3(self): return self.text_encoders.get("text_encoder_3")
    @property
    def _text_encoder_4(self): return self.text_encoders.get("text_encoder_4")
    @property
    def _tokenizer(self): return self.tokenizers.get("tokenizer")
    @property
    def _tokenizer_1(self): return self.tokenizers.get("tokenizer")
    @property
    def _tokenizer_2(self): return self.tokenizers.get("tokenizer_2")
    @property
    def _tokenizer_3(self): return self.tokenizers.get("tokenizer_3")
    @property
    def _tokenizer_4(self): return self.tokenizers.get("tokenizer_4")
    @property
    def _vae(self): return self.vae
    @property
    def _scheduler(self): return self.scheduler
    @property
    def _transformer_streamer(self): return self.transformer
    @property
    def _transformer(self): return self.transformer
    @property
    def _unet(self): return self.transformer
    @property
    def _vae_scale_factor(self):
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels"):
            return 2 ** (len(self.vae.config.block_out_channels) - 1)
        return 8

    def generate(self, prompt: str, **kwargs):
        module_name = f"weellm.generate.generate_{self.transformer_class_name}"
        try:
            import importlib
            gen_module = importlib.import_module(module_name)
        except ImportError:
            raise NotImplementedError(f"No generation logic implemented for {self.transformer_class_name}. Expected module {module_name}")
        
        return gen_module.generate(self, prompt, **kwargs)

    def free_text_encoder_ram(self):
        """Frees the RAM cache of all text encoders after encoding to save memory (they will lazy-reload if needed)."""
        import gc
        cleared = False
        for te_key in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"]:
            te = self.text_encoders.get(te_key)
            if te is not None:
                seeker = getattr(te, "seeker", getattr(te, "_seeker", None))
                if seeker is not None and hasattr(seeker, "clear_ram_cache"):
                    seeker.clear_ram_cache()
                    cleared = True
        if cleared:
            gc.collect()