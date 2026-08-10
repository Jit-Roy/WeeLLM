import os
import json
import importlib
import gc
from pathlib import Path
from typing import Union
import torch

class WeePipeline:
    """
    A unified entrypoint for creating native diffusers pipelines but with
    WeeLLM's memory-efficient layer streamers injected.
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
        
        # WeeLLM Auto-Dtype: bf16 is only hardware-accelerated on Ampere (SM >= 8.0) and above.
        # On older GPUs like Kaggle T4 (SM 7.5), bf16 runs in software, causing PyTorch to fall back 
        # to Math Attention which allocates a massive 4.5GB+ attention matrix.
        # Solution: silently downcast to float16 which older GPUs fully support in hardware.
        effective_dtype = torch_dtype
        if torch_dtype == torch.bfloat16 and torch.cuda.is_available():
            sm_major = torch.cuda.get_device_capability()[0]
            if sm_major < 8:
                effective_dtype = torch.float16
                print(f"  [WeeLLM] NOTE: bf16 is not natively supported on this GPU (SM {sm_major}.x < 8.0).")
                print(f"  [WeeLLM] Auto-casting to float16 to prevent VRAM spikes and ensure hardware-accelerated attention.\n")
        
        diffusers_kwargs["torch_dtype"] = effective_dtype

        # 1. Tokenizers & Scheduler
        print("[1/4] Loading Tokenizers and Scheduler ...")
        if "feature_extractor" in index:
            feature_extractor = None
            try:
                from transformers import AutoImageProcessor
                feature_extractor = AutoImageProcessor.from_pretrained(
                    str(model_dir), subfolder="feature_extractor"
                )
            except Exception:
                try:
                    from transformers import CLIPImageProcessor
                    feature_extractor = CLIPImageProcessor.from_pretrained(
                        str(model_dir), subfolder="feature_extractor"
                    )
                except Exception as e:
                    print(f"Warning: Failed to load feature_extractor, continuing with None: {e}")
            diffusers_kwargs["feature_extractor"] = feature_extractor

        if "safety_checker" in index:
            safety_checker = None
            try:
                from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
                safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                    str(model_dir), subfolder="safety_checker"
                )
            except Exception as e:
                print(f"Warning: Failed to load safety_checker, continuing with None: {e}")
            diffusers_kwargs["safety_checker"] = safety_checker

        for key in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
            if key in index:
                from transformers import AutoTokenizer
                try:
                    diffusers_kwargs[key] = AutoTokenizer.from_pretrained(str(model_dir), subfolder=key)
                except Exception as e:
                    print(f"Warning: Failed to load {key}: {e}")

        scheduler_cls = getattr(importlib.import_module("diffusers"), index["scheduler"][1])
        diffusers_kwargs["scheduler"] = scheduler_cls.from_pretrained(str(model_dir), subfolder="scheduler")
        
        # 2. VAE (Lazy Loaded on Meta Device)
        print("\n[2/4] Initializing VAE (Lazy loading on meta device) ...")
        from weellm.models.vaes.lazy_vae import LazyVAEStreamer
        vae_dir = model_dir / "vae"
        
        lazy_vae = LazyVAEStreamer.from_pretrained(
            vae_dir,
            device=device,
            dtype=torch_dtype,
            cache_to_ram=cache_to_ram
        )
        diffusers_kwargs["vae"] = lazy_vae.model

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
            "Qwen3VLModel": "weellm.models.text_encoders.qwen3_vl_model",
            "Mistral3ForConditionalGeneration": "weellm.models.text_encoders.mistral3_for_conditional_generation",
            "GlmModel": "weellm.models.text_encoders.glm_model",
            "Gemma2Model": "weellm.models.text_encoders.gemma2_model",
            "LlamaForCausalLM": "weellm.models.text_encoders.llama_for_causal_lm",
        }
        
        te_streamers = {}  # streamer references keyed by TE name, for post-encode eviction
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
                        
                    # Keep a reference to the streamer for post-encode eviction
                    te_streamers[key] = streamer
                    
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

        def _evict_module_to_meta(module, label: str):
            if module is None or not isinstance(module, torch.nn.Module):
                return
            try:
                evicted = 0
                # Evict parameters: replace .data with a meta-device tensor.
                # We avoid set_module_tensor_to_device for params because newer
                # PyTorch raises "invalid combination of arguments" when it tries
                # to call Tensor.new(..., requires_grad=bool) on the meta device.
                for name, param in list(module.named_parameters(recurse=True)):
                    dev = getattr(param, "device", None)
                    if dev is not None and dev.type != "meta":
                        try:
                            param.data = torch.empty(0, device="meta")
                        except Exception:
                            pass
                        evicted += 1

                # Evict buffers: use set_module_tensor_to_device (buffers don't
                # carry requires_grad so the API works fine here).
                from accelerate.utils.modeling import set_module_tensor_to_device
                for name, buf in module.named_buffers(recurse=True):
                    dev = getattr(buf, "device", None)
                    if dev is not None and dev.type != "meta":
                        try:
                            set_module_tensor_to_device(module, name, "meta")
                        except Exception:
                            pass

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WeeLLM Offload] Evicted {label} tensors to meta (params={evicted}).")
            except Exception as exc:
                print(f"[WeeLLM Offload] Failed to evict {label}: {exc}")

        if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
            te = pipeline.text_encoder
            print("[WeeLLM Debug] Pipeline text_encoder:", te.__class__.__name__)
            try:
                meta_params = sum(1 for p in te.parameters() if getattr(p, "device", None) is not None and p.device.type == "meta")
                meta_buffers = sum(1 for b in te.buffers() if getattr(b, "device", None) is not None and b.device.type == "meta")
                print(f"[WeeLLM Debug] text_encoder meta params={meta_params} meta buffers={meta_buffers}")
            except Exception as exc:
                print(f"[WeeLLM Debug] Unable to summarize text_encoder tensors: {exc}")

            if meta_params > 0:
                real_device = torch.device(device)
                pipeline_cls = pipeline.__class__

                if not hasattr(pipeline_cls, "_weellm_original_execution_device"):
                    pipeline_cls._weellm_original_execution_device = pipeline_cls._execution_device
                    pipeline_cls._weellm_original_device = pipeline_cls.device

                    def _forced_execution_device(self_obj):
                        return real_device

                    def _forced_device(self_obj):
                        return real_device

                    pipeline_cls._execution_device = property(_forced_execution_device)
                    pipeline_cls.device = property(_forced_device)
                    print(f"[WeeLLM Debug] Forced pipeline execution device to {real_device} because text_encoder still has meta weights.")


        try:
            execution_device = getattr(pipeline, "_execution_device", None)
            pipeline_device = getattr(pipeline, "device", None)
            print(f"[WeeLLM Debug] pipeline.device={pipeline_device} pipeline._execution_device={execution_device}")
        except Exception as exc:
            print(f"[WeeLLM Debug] Unable to inspect pipeline device properties: {exc}")

        def _move_scheduler_tensors_to_device(scheduler_obj, target_device):
            def _contains_tensor(value):
                if torch.is_tensor(value):
                    return True
                if isinstance(value, (list, tuple)):
                    return any(_contains_tensor(item) for item in value)
                if isinstance(value, dict):
                    return any(_contains_tensor(item) for item in value.values())
                return False

            def _move_value(value):
                if torch.is_tensor(value):
                    return value.to(target_device) if value.device.type != target_device else value
                if isinstance(value, list):
                    return [_move_value(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(_move_value(item) for item in value)
                if isinstance(value, dict):
                    return {key: _move_value(item) for key, item in value.items()}
                return value

            for attr_name, attr_value in list(vars(scheduler_obj).items()):
                if attr_name == "config":
                    continue
                if _contains_tensor(attr_value):
                    setattr(scheduler_obj, attr_name, _move_value(attr_value))

        if hasattr(pipeline, "scheduler"):
            _move_scheduler_tensors_to_device(pipeline.scheduler, device)
            if hasattr(pipeline.scheduler, "set_timesteps"):
                original_set_timesteps = pipeline.scheduler.set_timesteps
                original_step = pipeline.scheduler.step

                def safe_set_timesteps(self_obj, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
                    result = original_set_timesteps(
                        num_inference_steps=num_inference_steps,
                        device=device,
                        sigmas=sigmas,
                        mu=mu,
                        timesteps=timesteps,
                    )
                    _move_scheduler_tensors_to_device(self_obj, device)
                    return result

                def safe_step(self_obj, *args, **kwargs):
                    args = list(args)
                    tensor_device = None
                    for value in args[:3]:
                        if torch.is_tensor(value):
                            tensor_device = value.device.type
                            break
                    if tensor_device is None:
                        for key in ("model_output", "sample", "timestep"):
                            value = kwargs.get(key)
                            if torch.is_tensor(value):
                                tensor_device = value.device.type
                                break
                    if tensor_device is None:
                        tensor_device = device

                    _move_scheduler_tensors_to_device(self_obj, tensor_device)

                    for index, value in enumerate(args[:3]):
                        if torch.is_tensor(value) and value.device.type != tensor_device:
                            args[index] = value.to(tensor_device)
                    for key, value in list(kwargs.items()):
                        if torch.is_tensor(value) and value.device.type != tensor_device:
                            kwargs[key] = value.to(tensor_device)

                    result = original_step(*args, **kwargs)
                    _move_scheduler_tensors_to_device(self_obj, tensor_device)
                    return result

                import types
                pipeline.scheduler.set_timesteps = types.MethodType(safe_set_timesteps, pipeline.scheduler)
                pipeline.scheduler.step = types.MethodType(safe_step, pipeline.scheduler)
        
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
        print("\n[5/5] Applying Aggressive RAM Eviction...")
        
        # WeeLLM Auto-Optimization: Older GPUs (like Kaggle P100/T4) don't support Flash Attention 2,
        # which causes Math attention to allocate huge 4GB+ tensors.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("      -> [WeeLLM] Enabled xFormers memory-efficient attention for older GPU (Compute < 8.0).")
            except Exception:
                pass
        # pipeline.enable_model_cpu_offload(device=device) # Removed: this moves the VAE to CPU RAM, inflating RAM by 2GB!
        # Also enable VAE tiling to prevent massive decoding spikes
        try:
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
                
                # WeeLLM Aggressive Tiling: Force smaller tiles for massive VAEs (e.g. CogView4)
                pipeline.vae.tile_sample_min_size = 256
                if hasattr(pipeline.vae.config, "block_out_channels"):
                    pipeline.vae.tile_latent_min_size = int(
                        pipeline.vae.tile_sample_min_size / (2 ** (len(pipeline.vae.config.block_out_channels) - 1))
                    )
                
                # WeeLLM Aggressive Datatype: Disable float32 upcasting which doubles VRAM usage
                # if hasattr(pipeline.vae.config, "force_upcast") and pipeline.vae.config.force_upcast:
                #     pipeline.vae.config.force_upcast = False
                #     pipeline.vae.to(dtype=torch_dtype)
                #     print("      -> [WeeLLM] Disabled VAE float32 upcasting to save 50% memory.")
                
                print("      -> [WeeLLM] Enabled Aggressive VAE Tiling (via VAE) to prevent decoding VRAM spikes.")
                
            elif hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
                print("      -> [WeeLLM] Enabled VAE Tiling (via pipeline) to prevent decoding VRAM spikes.")
        except Exception:
            pass
        
        # 6. Aggressive One-Shot RAM Deletion for Kaggle
        if cache_to_ram:
            print("\n[6/6] Injecting Aggressive One-Shot RAM Eviction (cache_to_ram=True)...")
            import types
            import gc
            
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
        import gc
        
        def print_weellm_vram(tag):
            import torch, psutil, os
            try:
                alloc = torch.cuda.memory_allocated() / (1024**3)
                res = torch.cuda.memory_reserved() / (1024**3)
                ram = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                print(f"[WeeLLM VRAM Track] {tag:^45} | Alloc: {alloc:.3f} GB | Reserved: {res:.3f} GB | RAM: {ram:.3f} GB")
            except Exception:
                pass
        
        # Evict text encoders on the FIRST forward pass of the UNet/Transformer
        def _evict_te_before_unet(module, args):
            if getattr(module, "_weellm_te_evicted", False):
                return
            module._weellm_te_evicted = True
            
            print_weellm_vram("Before Text Encoder Offload")
            for te_key, streamer_obj in te_streamers.items():
                if hasattr(streamer_obj, "evict_resident"):
                    try:
                        streamer_obj.evict_resident()
                        print(f"[WeeLLM Offload] Evicted {te_key} resident weights to meta.")
                    except Exception as exc:
                        print(f"[WeeLLM Offload] Failed to evict {te_key} resident: {exc}")
                else:
                    te_mod = getattr(pipeline, te_key, None)
                    _evict_module_to_meta(te_mod, te_key)
            
            if cache_to_ram:
                print("\n[WeeLLM] One-Shot: Freeing Text Encoders from RAM...")
                for te_name in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
                    if hasattr(pipeline, te_name):
                        setattr(pipeline, te_name, None)
                        
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_weellm_vram("After Text Encoder Offload")

        if hasattr(pipeline, "unet") and pipeline.unet is not None:
            pipeline.unet.register_forward_pre_hook(_evict_te_before_unet)
        elif hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            pipeline.transformer.register_forward_pre_hook(_evict_te_before_unet)
            
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
            original_vae_decode_vram = pipeline.vae.decode
            def defrag_vae_decode(self_obj, *args, **kwargs):
                print_weellm_vram("Before VAE Decode (Before GC)")
                gc.collect()
                torch.cuda.empty_cache()
                print_weellm_vram("Before VAE Decode (After GC)")

                # Denoising is done; evict transformer before lazy VAE load.
                _evict_module_to_meta(getattr(pipeline, "transformer", None), "transformer")
                print_weellm_vram("Before VAE Decode (After Transformer Offload)")

                res = original_vae_decode_vram(*args, **kwargs)
                print_weellm_vram("After VAE Decode (Before GC)")
                gc.collect()
                torch.cuda.empty_cache()
                print_weellm_vram("After VAE Decode (After GC)")
                return res
            pipeline.vae.decode = types.MethodType(defrag_vae_decode, pipeline.vae)
            
        return pipeline
