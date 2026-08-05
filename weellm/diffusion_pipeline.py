import os
import json
import importlib
from pathlib import Path
from typing import Union
import torch

class DiffusionPipeline:
    """
    Factory class that returns a native Hugging Face pipeline armed with WeeLLM streamers.
    """
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
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
            
        pipeline_class_name = index.get('_class_name')
        if not pipeline_class_name:
            raise ValueError("No _class_name found in model_index.json")

        print("\n============================================================")
        print(f"  WeeLLM -- Building Native {pipeline_class_name} with Streamers")
        print("============================================================\n")
        
        diffusers_kwargs = kwargs.copy()
        diffusers_kwargs["torch_dtype"] = torch_dtype

        # 1. Tokenizers & Scheduler
        print("[1/4] Loading Tokenizers and Scheduler ...")
        for key in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
            if key in index:
                from transformers import AutoTokenizer
                try:
                    diffusers_kwargs[key] = AutoTokenizer.from_pretrained(str(model_dir), subfolder=key)
                except Exception as e:
                    print(f"Warning: Failed to load {key}: {e}")

        scheduler_cls = getattr(importlib.import_module("diffusers"), index["scheduler"][1])
        diffusers_kwargs["scheduler"] = scheduler_cls.from_pretrained(str(model_dir), subfolder="scheduler")
        
        # 2. VAE (Resident on GPU)
        print("\n[2/4] Loading VAE (resident on GPU) ...")
        vae_cls = getattr(importlib.import_module("diffusers"), index["vae"][1])
        vae_dir = model_dir / "vae"
        has_fp16 = (vae_dir / "diffusion_pytorch_model.fp16.safetensors").exists()
        
        if has_fp16:
            diffusers_kwargs["vae"] = vae_cls.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=torch_dtype, variant="fp16", use_safetensors=True).to(device)
        else:
            diffusers_kwargs["vae"] = vae_cls.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=torch_dtype, use_safetensors=True).to(device)

        # 3. Text Encoders (Streamed or Resident)
        print("\n[3/4] Preparing Text Encoders ...")
        TE_MAP = {
            "CLIPTextModel": "weellm.models.text_encoders.clip_text_model",
            "CLIPTextModelWithProjection": "weellm.models.text_encoders.clip_text_model",
            "T5EncoderModel": "weellm.models.text_encoders.t5_encoder_model",
            "UMT5EncoderModel": "weellm.models.text_encoders.umt5_encoder_model",
            "Qwen2ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen3ForCausalLM": "weellm.models.text_encoders.qwen3_for_causal_lm",
            "Qwen2_5_VLForConditionalGeneration": "weellm.models.text_encoders.qwen2_5_vl_for_conditional_generation",
            "Qwen3VLModel": "weellm.models.text_encoders.qwen3_vl_model",
            "Mistral3ForConditionalGeneration": "weellm.models.text_encoders.mistral3_for_conditional_generation",
            "GlmModel": "weellm.models.text_encoders.glm_model",
            "Gemma2Model": "weellm.models.text_encoders.gemma2_model",
            "LlamaForCausalLM": "weellm.models.text_encoders.llama_for_causal_lm",
        }
        
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
                        # Some CausalLM streamers just take text_encoder_dir directly
                        if hasattr(te_cls, "from_pretrained"):
                            streamer = te_cls.from_pretrained(model_dir=te_path, tokenizer=diffusers_kwargs.get(tok_key), device=device, dtype=torch_dtype, cache_to_ram=cache_to_ram)
                            if hasattr(streamer, "_ensure_initialized"):
                                streamer._ensure_initialized()
                        else:
                            # Direct instantiation
                            streamer = te_cls(text_encoder_dir=te_path, tokenizer_dir=str(model_dir / tok_key), device=device, dtype=torch_dtype, cache_to_ram=cache_to_ram)
                            if hasattr(streamer, "_ensure_initialized"):
                                streamer._ensure_initialized()
                    elif "CLIP" in hf_cls_name:
                        hf_module = importlib.import_module("transformers")
                        hf_cls = getattr(hf_module, hf_cls_name)
                        streamer = te_cls.from_pretrained(hf_cls, str(model_dir), key, device=device, dtype=torch_dtype, output_hidden_states=True, cache_to_ram=cache_to_ram)
                    else:
                        streamer = te_cls.from_pretrained(model_dir=te_path, device=device, dtype=torch_dtype, cache_to_ram=cache_to_ram)
                        if hasattr(streamer, "_ensure_initialized"):
                            streamer._ensure_initialized()
                        
                    # Inject the actual model module directly for the diffusers pipeline
                    te_model = getattr(streamer, "model", getattr(streamer, "_model", streamer))
                    
                    # Patch .to() via dynamic subclassing so accelerate doesn't delete it on hook removal
                    if hasattr(te_model, "to"):
                        class PatchedTEModel(te_model.__class__):
                            def to(self_obj, *args, **kwargs):
                                def safe_convert(t):
                                    if t.device.type != "meta":
                                        return t.to(*args, **kwargs)
                                    return t
                                return self_obj._apply(safe_convert)
                                
                        PatchedTEModel.__name__ = te_model.__class__.__name__
                        PatchedTEModel.__qualname__ = getattr(te_model.__class__, "__qualname__", te_model.__class__.__name__)
                        PatchedTEModel.__module__ = te_model.__class__.__module__
                        te_model.__class__ = PatchedTEModel
                            
                    diffusers_kwargs[key] = te_model
                else:
                    hf_module = importlib.import_module("transformers")
                    hf_cls = getattr(hf_module, hf_cls_name)
                    diffusers_kwargs[key] = hf_cls.from_pretrained(str(model_dir), subfolder=key, torch_dtype=torch_dtype).to(device)
                    diffusers_kwargs[key].eval()

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
            "Ideogram4Transformer2DModel": "weellm.models.transformers.ideogram4_transformer",
            "UNet2DConditionModel": "weellm.models.unets.unet_2d_condition_model"
        }
        
        module_path = TR_MAP.get(transformer_class_name, "")
        if not module_path:
            raise ValueError(f"Unsupported architecture: {transformer_class_name}")
            
        module = importlib.import_module(module_path)
        transformer_cls_streamer = getattr(module, transformer_class_name + "Streamer")
        
        if transformer_key == "unet":
            transformer_streamer = transformer_cls_streamer.from_pretrained(str(model_dir), device, torch_dtype, prefetch, cache_to_ram=cache_to_ram)
        else:
            transformer_streamer = transformer_cls_streamer.from_pretrained(model_dir / transformer_key, device=device, dtype=torch_dtype, prefetch=prefetch, cache_to_ram=cache_to_ram)
            
        if hasattr(transformer_streamer, "_ensure_initialized"):
            transformer_streamer._ensure_initialized()
        
        
        # Inject the actual model module directly for the diffusers pipeline
        tr_model = getattr(transformer_streamer, "model", getattr(transformer_streamer, "_model", transformer_streamer))
        
        # Patch .to() via dynamic subclassing so accelerate doesn't delete it on hook removal
        if hasattr(tr_model, "to"):
            class PatchedTRModel(tr_model.__class__):
                def to(self_obj, *args, **kwargs):
                    def safe_convert(t):
                        if t.device.type != "meta":
                            return t.to(*args, **kwargs)
                        return t
                    return self_obj._apply(safe_convert)
                    
            PatchedTRModel.__name__ = tr_model.__class__.__name__
            PatchedTRModel.__qualname__ = getattr(tr_model.__class__, "__qualname__", tr_model.__class__.__name__)
            PatchedTRModel.__module__ = tr_model.__class__.__module__
            tr_model.__class__ = PatchedTRModel
            
        diffusers_kwargs[transformer_key] = tr_model
        
        # Some pipelines (e.g. Ideogram4) require an unconditional_transformer
        if "unconditional_transformer" in index:
            diffusers_kwargs["unconditional_transformer"] = tr_model
        
        print("\n============================================================")
        print("  Instantiating Native Diffusers Pipeline ...")
        print("============================================================\n")
        
        diffusers_kwargs.pop("torch_dtype", None)
        
        pipeline_module = importlib.import_module("diffusers")
        pipeline_cls = getattr(pipeline_module, pipeline_class_name)
        
        pipeline = pipeline_cls(**diffusers_kwargs)
        
        import types
        original_pipeline_to = pipeline.to
        
        def safe_pipeline_to(self_obj, *args, **kwargs):
            # Temporarily hide modules that contain meta tensors so diffusers' original .to() ignores them
            hidden_meta_modules = {}
            for name, module in list(self_obj.components.items()):
                if isinstance(module, torch.nn.Module):
                    has_meta_param = any(p.device.type == "meta" for p in getattr(module, "parameters", lambda: [])())
                    has_meta_buffer = any(b.device.type == "meta" for b in getattr(module, "buffers", lambda: [])())
                    if has_meta_param or has_meta_buffer:
                        hidden_meta_modules[name] = module
                        setattr(self_obj, name, None)  # self_obj.components is a dynamic property!
            try:
                # Call original pipeline.to() which handles diffusers-specific kwargs (e.g. silence_dtype_warnings)
                res = original_pipeline_to(*args, **kwargs)
            finally:
                # Restore the meta modules
                for name, module in hidden_meta_modules.items():
                    setattr(self_obj, name, module)
            return res
            
        pipeline.to = types.MethodType(safe_pipeline_to, pipeline)
        

        
        # 5. Apply the Aggressive RAM Eviction Optimization
        print("\n[5/5] Applying Aggressive RAM Eviction (Model CPU Offload)...")
        pipeline.enable_model_cpu_offload(device=device)
        
        # 6. Aggressive One-Shot RAM Deletion for Kaggle
        if cache_to_ram:
            print("\n[6/6] Injecting Aggressive One-Shot RAM Eviction (cache_to_ram=True)...")
            import types
            import gc
            
            # Patch encode_prompt to destroy text encoders immediately after
            if hasattr(pipeline, "encode_prompt"):
                original_encode_prompt = pipeline.encode_prompt
                
                def aggressive_encode_prompt(self_obj, *args, **kwargs):
                    res = original_encode_prompt(*args, **kwargs)
                    print("\n[WeeLLM] One-Shot: Freeing Text Encoders from RAM...")
                    for te_name in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
                        if hasattr(self_obj, te_name):
                            setattr(self_obj, te_name, None)
                    gc.collect()
                    return res
                    
                pipeline.encode_prompt = types.MethodType(aggressive_encode_prompt, pipeline)
            
            # Patch VAE decode to destroy transformer immediately before decoding
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
                original_vae_decode = pipeline.vae.decode
                
                def aggressive_vae_decode(self_obj, *args, **kwargs):
                    print("\n[WeeLLM] One-Shot: Freeing Transformer from RAM before VAE Decode...")
                    for tr_name in ["transformer", "unet"]:
                        if hasattr(pipeline, tr_name):
                            setattr(pipeline, tr_name, None)
                    gc.collect()
                    return original_vae_decode(*args, **kwargs)
                    
                pipeline.vae.decode = types.MethodType(aggressive_vae_decode, pipeline.vae)
                
        # 7. Aggressive VRAM Defragmentation
        import types
        
        def print_weellm_vram(tag):
            import torch, psutil, os
            try:
                alloc = torch.cuda.memory_allocated() / (1024**3)
                res = torch.cuda.memory_reserved() / (1024**3)
                ram = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                print(f"[WeeLLM VRAM Track] {tag:^45} | Alloc: {alloc:.3f} GB | Reserved: {res:.3f} GB | RAM: {ram:.3f} GB")
            except Exception:
                pass
        
        if hasattr(pipeline, "encode_prompt"):
            original_encode_prompt_vram = pipeline.encode_prompt
            def defrag_encode_prompt(self_obj, *args, **kwargs):
                print_weellm_vram("Before Text Encoder")
                res = original_encode_prompt_vram(*args, **kwargs)
                print_weellm_vram("After Text Encoder (Before GC)")
                import gc; gc.collect()
                torch.cuda.empty_cache()
                print_weellm_vram("After Text Encoder (After GC)")
                return res
            pipeline.encode_prompt = types.MethodType(defrag_encode_prompt, pipeline)
            
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
            original_vae_decode_vram = pipeline.vae.decode
            def defrag_vae_decode(self_obj, *args, **kwargs):
                print_weellm_vram("Before VAE Decode (Before GC)")
                import gc; gc.collect()
                torch.cuda.empty_cache()
                print_weellm_vram("Before VAE Decode (After GC)")
                res = original_vae_decode_vram(*args, **kwargs)
                print_weellm_vram("After VAE Decode (Before GC)")
                import gc; gc.collect()
                torch.cuda.empty_cache()
                print_weellm_vram("After VAE Decode (After GC)")
                return res
            pipeline.vae.decode = types.MethodType(defrag_vae_decode, pipeline.vae)
            
        return pipeline
