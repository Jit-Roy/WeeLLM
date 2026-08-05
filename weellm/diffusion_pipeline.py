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

            if te.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
                import types

                original_forward = te.forward

                def debug_qwen_forward(self_obj, *args, **kwargs):
                    print("[WeeLLM Qwen Debug] Qwen2_5_VLForConditionalGeneration.forward called")
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "past_key_values",
                        "inputs_embeds",
                        "image_grid_thw",
                        "video_grid_thw",
                        "second_per_grid_ts",
                        "rope_deltas",
                        "cache_position",
                    ):
                        value = kwargs.get(key)
                        if torch.is_tensor(value):
                            print(f"[WeeLLM Qwen Debug] kwargs[{key}] shape={tuple(value.shape)} device={value.device} dtype={value.dtype}")
                        elif value is not None:
                            print(f"[WeeLLM Qwen Debug] kwargs[{key}] type={type(value).__name__}")
                    try:
                        rope_deltas = getattr(self_obj.model, "rope_deltas", None)
                        if torch.is_tensor(rope_deltas):
                            print(f"[WeeLLM Qwen Debug] self.model.rope_deltas shape={tuple(rope_deltas.shape)} device={rope_deltas.device} dtype={rope_deltas.dtype}")
                        else:
                            print(f"[WeeLLM Qwen Debug] self.model.rope_deltas={type(rope_deltas).__name__ if rope_deltas is not None else 'None'}")
                    except Exception as exc:
                        print(f"[WeeLLM Qwen Debug] Unable to inspect rope_deltas: {exc}")

                    if kwargs.get("pixel_values") is None and kwargs.get("pixel_values_videos") is None:
                        print("[WeeLLM Qwen Debug] Using pure-text bypass through inner language_model; lm_head is skipped.")
                        inner_kwargs = {
                            "input_ids": kwargs.get("input_ids"),
                            "attention_mask": kwargs.get("attention_mask"),
                            "position_ids": kwargs.get("position_ids"),
                            "past_key_values": kwargs.get("past_key_values"),
                            "inputs_embeds": kwargs.get("inputs_embeds"),
                            "use_cache": False,
                            "output_attentions": kwargs.get("output_attentions", False),
                            "output_hidden_states": True,
                            "return_dict": True,
                            "cache_position": kwargs.get("cache_position"),
                        }
                        inner_kwargs = {k: v for k, v in inner_kwargs.items() if v is not None}
                        inner_out = self_obj.model.language_model(**inner_kwargs)
                        try:
                            if hasattr(inner_out, "hidden_states") and inner_out.hidden_states is not None:
                                print(f"[WeeLLM Qwen Debug] bypass hidden_states count={len(inner_out.hidden_states)}")
                                final_hidden = inner_out.hidden_states[-1]
                                print(
                                    f"[WeeLLM Qwen Debug] bypass final hidden shape={tuple(final_hidden.shape)} "
                                    f"device={final_hidden.device} dtype={final_hidden.dtype}"
                                )
                        except Exception as exc:
                            print(f"[WeeLLM Qwen Debug] bypass output inspect failed: {exc}")

                        import types as _types

                        return _types.SimpleNamespace(
                            hidden_states=inner_out.hidden_states,
                            past_key_values=getattr(inner_out, "past_key_values", None),
                            attentions=getattr(inner_out, "attentions", None),
                            logits=None,
                        )

                    return original_forward(*args, **kwargs)

                te.forward = types.MethodType(debug_qwen_forward, te)

                try:
                    from transformers import masking_utils

                    if not hasattr(masking_utils, "_weellm_original_create_causal_mask"):
                        masking_utils._weellm_original_create_causal_mask = masking_utils.create_causal_mask

                        def debug_create_causal_mask(*args, **kwargs):
                            print("[WeeLLM Qwen Debug] create_causal_mask called")
                            for key in ("input_embeds", "attention_mask", "cache_position", "past_key_values", "position_ids"):
                                value = kwargs.get(key)
                                if torch.is_tensor(value):
                                    print(f"[WeeLLM Qwen Debug] mask_kwargs[{key}] shape={tuple(value.shape)} device={value.device} dtype={value.dtype}")
                                elif value is not None:
                                    print(f"[WeeLLM Qwen Debug] mask_kwargs[{key}] type={type(value).__name__}")
                            return masking_utils._weellm_original_create_causal_mask(*args, **kwargs)

                        masking_utils.create_causal_mask = debug_create_causal_mask
                except Exception as exc:
                    print(f"[WeeLLM Qwen Debug] Unable to patch create_causal_mask: {exc}")

        try:
            execution_device = getattr(pipeline, "_execution_device", None)
            pipeline_device = getattr(pipeline, "device", None)
            print(f"[WeeLLM Debug] pipeline.device={pipeline_device} pipeline._execution_device={execution_device}")
        except Exception as exc:
            print(f"[WeeLLM Debug] Unable to inspect pipeline device properties: {exc}")

        if hasattr(pipeline, "_get_qwen_prompt_embeds"):
            original_get_qwen_prompt_embeds = pipeline._get_qwen_prompt_embeds

            def debug_get_qwen_prompt_embeds(self_obj, *args, **kwargs):
                print("[WeeLLM Qwen Debug] _get_qwen_prompt_embeds entered")
                if args:
                    print(f"[WeeLLM Qwen Debug] positional args count={len(args)} first={args[0]!r}")
                for key in ("prompt", "device", "dtype"):
                    value = kwargs.get(key)
                    if value is not None:
                        print(f"[WeeLLM Qwen Debug] kwarg {key}={value!r}")
                try:
                    print(
                        f"[WeeLLM Qwen Debug] pipeline._execution_device at prompt encode={getattr(self_obj, '_execution_device', None)}"
                    )
                except Exception as exc:
                    print(f"[WeeLLM Qwen Debug] unable to read _execution_device inside prompt embedder: {exc}")
                result = original_get_qwen_prompt_embeds(*args, **kwargs)
                try:
                    if isinstance(result, tuple) and len(result) >= 2:
                        prompt_embeds, prompt_mask = result[:2]
                        target_dtype = getattr(getattr(self_obj, "transformer", None), "dtype", None)
                        if target_dtype is None:
                            target_dtype = getattr(getattr(self_obj, "text_encoder", None), "dtype", None)
                        if target_dtype is None:
                            target_dtype = torch.bfloat16
                        if torch.is_tensor(prompt_embeds) and prompt_embeds.dtype != target_dtype:
                            print(
                                f"[WeeLLM Qwen Debug] casting prompt_embeds {prompt_embeds.dtype} -> {target_dtype} "
                                f"to keep latent/transformer dtypes aligned"
                            )
                            prompt_embeds = prompt_embeds.to(dtype=target_dtype)
                        print(f"[WeeLLM Qwen Debug] _get_qwen_prompt_embeds output embeds shape={tuple(prompt_embeds.shape)} device={prompt_embeds.device} dtype={prompt_embeds.dtype}")
                        print(f"[WeeLLM Qwen Debug] _get_qwen_prompt_embeds output mask shape={tuple(prompt_mask.shape)} device={prompt_mask.device} dtype={prompt_mask.dtype}")
                        result = (prompt_embeds, prompt_mask)
                except Exception as exc:
                    print(f"[WeeLLM Qwen Debug] unable to inspect prompt embed outputs: {exc}")
                return result

            pipeline._get_qwen_prompt_embeds = types.MethodType(debug_get_qwen_prompt_embeds, pipeline)

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
                if hasattr(pipeline.vae.config, "force_upcast") and pipeline.vae.config.force_upcast:
                    pipeline.vae.config.force_upcast = False
                    pipeline.vae.to(dtype=torch_dtype)
                    print("      -> [WeeLLM] Disabled VAE float32 upcasting to save 50% memory.")
                
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
