"""
pipeline.py -- WeePipeline: Universal entrypoint for WeeLLM memory-efficient inference.

Builds native diffusers pipelines with WeeLLM layer-streamers injected for
every major component (VAE, text encoders, transformer/UNet). Applies
optimizations (VAE tiling, xformers, scheduler patching, meta-device eviction)
transparently.

``WeePipeline.from_pretrained()`` returns a :class:`WeePipeline` instance that
wraps the diffusers pipeline.  The wrapper supports:

* ``pipe(...)``              — direct diffusers pipeline call
* ``pipe.generate(...)``    — convenience wrapper (returns first image)
* ``pipe.<attr>``           — transparent attribute delegation to the inner pipeline
"""

import gc
import importlib
import json
import logging
import types
from pathlib import Path
from typing import Optional, Union

import torch

from weellm.utils import clean_memory, report_memory, resolve_model_path
from weellm.memory import evict_module

logger = logging.getLogger("weellm")

# ---------------------------------------------------------------------------
# Text encoder class → module path mapping
# ---------------------------------------------------------------------------
_TE_MAP = {
    "CLIPTextModel":                          "weellm.models.text_encoders.clip_text_model",
    "CLIPTextModelWithProjection":            "weellm.models.text_encoders.clip_text_model",
    "T5EncoderModel":                         "weellm.models.text_encoders.t5_encoder_model",
    "UMT5EncoderModel":                       "weellm.models.text_encoders.umt5_encoder_model",
    "Qwen2ForCausalLM":                       "weellm.models.text_encoders.qwen3_for_causal_lm",
    "Qwen3ForCausalLM":                       "weellm.models.text_encoders.qwen3_for_causal_lm",
    "Qwen3Model":                             "weellm.models.text_encoders.qwen3_for_causal_lm",
    "Qwen2_5_VLForConditionalGeneration":     "weellm.models.text_encoders.qwen2_5_vl_for_conditional_generation",
    "Qwen3VLModel":                           "weellm.models.text_encoders.qwen3_vl_model",
    "Mistral3ForConditionalGeneration":       "weellm.models.text_encoders.mistral3_for_conditional_generation",
    "GlmModel":                               "weellm.models.text_encoders.glm_model",
    "Gemma2Model":                            "weellm.models.text_encoders.gemma2_model",
    "LlamaForCausalLM":                       "weellm.models.text_encoders.llama_for_causal_lm",
}

# ---------------------------------------------------------------------------
# External repo fallback map
#
# Some pipelines reference a text encoder by class name in model_index.json
# but do NOT ship the weights inside the repo (e.g. HiDream references
# LlamaForCausalLM but the weights live in meta-llama/Meta-Llama-3.1-8B-Instruct).
# When the local subfolder is missing we fall back to downloading from here.
# ---------------------------------------------------------------------------
_TE_EXTERNAL_REPO = {
    "LlamaForCausalLM": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# ---------------------------------------------------------------------------
# Transformer class → module path mapping
# ---------------------------------------------------------------------------
_TR_MAP = {
    "FluxTransformer2DModel":              "weellm.models.transformers.flux_transformer_2d_model",
    "Flux2Transformer2DModel":             "weellm.models.transformers.flux2_transformer_2d_model",
    "ZImageTransformer2DModel":            "weellm.models.transformers.z_image_transformer_2d_model",
    "SD3Transformer2DModel":               "weellm.models.transformers.sd3_transformer_2d_model",
    "QwenImageTransformer2DModel":         "weellm.models.transformers.qwen_image_transformer_2d_model",
    "CogView4Transformer2DModel":          "weellm.models.transformers.cogview4_transformer_2d_model",
    "Lumina2Transformer2DModel":           "weellm.models.transformers.lumina2_transformer_2d_model",
    "AuraFlowTransformer2DModel":          "weellm.models.transformers.auraflow_transformer_2d_model",
    "HiDreamImageTransformer2DModel":      "weellm.models.transformers.hidream_transformer_2d_model",
    "Ideogram4Transformer2DModel":         "weellm.models.transformers.ideogram4_transformer",
    "UNet2DConditionModel":                "weellm.models.unets.unet_2d_condition_model",
}


# ---------------------------------------------------------------------------
# WeePipeline wrapper
# ---------------------------------------------------------------------------

class WeePipeline:
    """
    A unified wrapper around a native diffusers pipeline that has WeeLLM
    memory-efficient layer-streamers injected into every major component.

    Use :meth:`from_pretrained` to construct — do **not** instantiate directly.

    The wrapper is transparent: attribute access and ``__call__`` are delegated
    to the underlying diffusers pipeline, so all existing diffusers code works
    unchanged.  Additionally, the :meth:`generate` convenience method is
    available as an instance method.

    Example::

        pipe = WeePipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

        # Text-to-image
        image = pipe.generate("A lion at sunset", seed=42)
        image.save("output.png")

        # Image-to-image (pass an img2img pipeline or use a native img2img model)
        image2 = pipe.generate(
            "A cyberpunk lion at sunset",
            image=image,
            strength=0.75,
            num_inference_steps=20,
            seed=0,
        )
        image2.save("output_img2img.png")

        # Direct diffusers call — fully supported
        out = pipe(prompt="A lion at sunset", num_inference_steps=4)
        out.images[0].save("direct.png")
    """

    def __init__(self, pipeline) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ during init.
        object.__setattr__(self, "_pipeline", pipeline)

    # ------------------------------------------------------------------
    # Transparent delegation
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Forward all calls directly to the underlying diffusers pipeline."""
        return self._pipeline(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to the inner diffusers pipeline."""
        return getattr(self._pipeline, name)

    def __setattr__(self, name: str, value):
        if name == "_pipeline":
            object.__setattr__(self, name, value)
        else:
            setattr(self._pipeline, name, value)

    def __repr__(self) -> str:
        return f"WeePipeline({self._pipeline.__class__.__name__})"

    # ------------------------------------------------------------------
    # Convenience generate() — instance method
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs):
        """
        Convenience wrapper that calls the pipeline and returns the first image.

        Supports all standard diffusers kwargs, including image-to-image and
        inpainting parameters::

            pipe.generate("A lion", seed=42)
            pipe.generate("A lion", image=pil_img, strength=0.75, num_inference_steps=20)
            pipe.generate("A lion", image=pil_img, mask_image=pil_mask)

        Parameters
        ----------
        prompt:
            Text prompt for image generation.
        seed:
            Optional integer random seed (extracted from kwargs).
        **kwargs:
            Any additional arguments forwarded to the diffusers pipeline call
            (e.g. ``height``, ``width``, ``num_inference_steps``, ``guidance_scale``,
            ``image``, ``mask_image``, ``strength``, ``negative_prompt`` …).

        Returns
        -------
        ``PIL.Image.Image`` — the first generated image.
        """
        seed = kwargs.pop("seed", None)
        generator: Optional[torch.Generator] = None
        if seed is not None:
            device = getattr(self._pipeline, "device", torch.device("cpu"))
            generator = torch.Generator(device=device).manual_seed(seed)

        out = self._pipeline(prompt=prompt, generator=generator, **kwargs)
        if hasattr(out, "images"):
            return out.images[0]
        return out[0][0]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
        vae_tile_size: int = 256,
        **kwargs,
    ) -> "WeePipeline":
        """
        Build a native diffusers pipeline with WeeLLM streamers injected.

        Parameters
        ----------
        model_dir:
            Hugging Face repo ID (e.g. ``"black-forest-labs/FLUX.1-schnell"``)
            or local directory containing ``model_index.json``.
        device:
            PyTorch device string (default: ``"cuda"``).
            ``"cpu"`` is supported on Windows/Linux; Apple Silicon (MPS) is not.
        torch_dtype:
            Compute dtype. Auto-downcast to float32 on GPUs that lack bfloat16
            support; auto-upcast float32→bfloat16 when the GPU natively supports it.
        prefetch:
            Enable background prefetching of the next layer while the current
            one is computing (default: ``True``).
        cache_to_ram:
            Load full safetensors shards into CPU RAM before serving tensors.
            Faster on slow cloud drives (Kaggle/Colab), uses more RAM.
        vae_tile_size:
            Minimum tile size (in pixels) for aggressive VAE tiling.
            Smaller values reduce VRAM spikes at the cost of slight quality
            degradation at tile boundaries (default: 256).

        Returns
        -------
        :class:`WeePipeline` wrapping the native diffusers pipeline, ready for
        ``pipe(...)`` or ``pipe.generate(...)`` calls.
        """
        model_dir_str  = str(resolve_model_path(str(model_dir)))
        model_dir_path = Path(model_dir_str)

        index = cls._load_index(model_dir_path)
        pipeline_class_name = index.get("_class_name")
        if not pipeline_class_name:
            raise ValueError("No _class_name found in model_index.json")

        logger.info("\n============================================================")
        logger.info("  WeeLLM -- Building Native %s with Streamers", pipeline_class_name)
        logger.info("============================================================\n")

        effective_dtype  = cls._resolve_dtype(device, torch_dtype)
        diffusers_kwargs = dict(kwargs)
        diffusers_kwargs["torch_dtype"] = effective_dtype

        # ── Step 1: Tokenizers & Scheduler ──────────────────────────────
        logger.info("[1/4] Loading Tokenizers and Scheduler ...")
        cls._load_tokenizers_and_scheduler(model_dir_path, index, device, effective_dtype, diffusers_kwargs)

        # ── Step 2: VAE ─────────────────────────────────────────────────
        logger.info("\n[2/4] Initializing VAE (Lazy loading on meta device) ...")

        # VAEs often produce artifacts in half-precision due to intermediate activation overflow.
        # If we are running in float16 or bfloat16, upcast the VAE to float32.
        vae_dtype = torch.float32 if effective_dtype in (torch.float16, torch.bfloat16) else effective_dtype
        lazy_vae  = cls._load_vae(model_dir_path, device, vae_dtype, cache_to_ram)
        diffusers_kwargs["vae"] = lazy_vae.model

        # ── Step 3: Text Encoders ────────────────────────────────────────
        logger.info("\n[3/4] Preparing Text Encoders ...")
        te_streamers = cls._load_text_encoders(
            model_dir_path, index, device, effective_dtype, effective_dtype, cache_to_ram, diffusers_kwargs
        )

        # ── Step 4: Transformer / UNet ──────────────────────────────────
        logger.info("\n[4/4] Preparing Transformer / UNet ...")
        transformer_key, transformer_streamer = cls._load_transformer(
            model_dir_path, index, device, effective_dtype, prefetch, cache_to_ram
        )
        tr_model = getattr(transformer_streamer, "model", getattr(transformer_streamer, "_model", transformer_streamer))
        tr_model = cls._patch_to(tr_model)
        diffusers_kwargs[transformer_key] = tr_model

        if "unconditional_transformer" in index:
            diffusers_kwargs["unconditional_transformer"] = tr_model

        # ── Instantiate pipeline ─────────────────────────────────────────
        logger.info("\n============================================================")
        logger.info("  Instantiating Native Diffusers Pipeline ...")
        logger.info("============================================================\n")

        diffusers_kwargs.pop("torch_dtype", None)
        pipeline_module = importlib.import_module("diffusers")
        pipeline_cls    = getattr(pipeline_module, pipeline_class_name)
        pipeline        = pipeline_cls(**diffusers_kwargs)

        # ── Post-build patches ───────────────────────────────────────────
        cls._patch_execution_device(pipeline, device, te_streamers)
        cls._patch_scheduler(pipeline, device)
        cls._patch_pipeline_to(pipeline)
        cls._apply_optimizations(pipeline, device, cache_to_ram, te_streamers, transformer_key, vae_tile_size)

        return cls(pipeline)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_index(model_dir: Path) -> dict:
        index_path = model_dir / "model_index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _resolve_dtype(device: str, torch_dtype: torch.dtype) -> torch.dtype:
        """Smart dtype resolution.

        * bfloat16 on a GPU that does not support it → float32 (safe fallback)
        * float32 is always honoured as-is — no silent upcast.
        * CPU: bfloat16 is supported natively since PyTorch ≥1.10; keep as-is.
        """
        if device == "cpu" or not torch.cuda.is_available():
            return torch_dtype

        bf16_supported = torch.cuda.is_bf16_supported()

        if torch_dtype == torch.bfloat16 and not bf16_supported:
            logger.info(
                "  [WeeLLM] NOTE: bfloat16 is not supported on this GPU.\n"
                "  [WeeLLM] Auto-casting to float32 (safe fallback) to prevent NaNs/black images "
                "that can occur with float16 on large models.\n"
            )
            return torch.float32

        if torch_dtype == torch.float32:
            logger.info(
                "  [WeeLLM] NOTE: float32 requested. Running in full precision.\n"
                "  [WeeLLM] Memory budget is tighter — use --dtype bfloat16 for ~2× speedup "
                "with equivalent quality.\n"
            )

        return torch_dtype

    @staticmethod
    def _load_tokenizers_and_scheduler(
        model_dir: Path, index: dict, device: str, effective_dtype: torch.dtype, out: dict
    ) -> None:
        """Populate *out* with tokenizers, feature_extractor, safety_checker, scheduler."""
        if "feature_extractor" in index:
            fe = None
            try:
                from transformers import AutoImageProcessor
                fe = AutoImageProcessor.from_pretrained(str(model_dir), subfolder="feature_extractor")
            except Exception:
                try:
                    from transformers import CLIPImageProcessor
                    fe = CLIPImageProcessor.from_pretrained(str(model_dir), subfolder="feature_extractor")
                except Exception as e:
                    logger.warning("Failed to load feature_extractor, continuing with None: %s", e)
            out["feature_extractor"] = fe

        if "safety_checker" in index:
            sc = None
            try:
                from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
                sc = StableDiffusionSafetyChecker.from_pretrained(
                    str(model_dir), subfolder="safety_checker", torch_dtype=effective_dtype
                )
                if sc is not None:
                    sc = sc.to(device)
            except Exception as e:
                logger.warning("Failed to load safety_checker, continuing with None: %s", e)
            out["safety_checker"] = sc

        for key in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
            if key in index:
                try:
                    from transformers import AutoTokenizer
                    out[key] = AutoTokenizer.from_pretrained(str(model_dir), subfolder=key)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", key, e)

        scheduler_cls = getattr(
            importlib.import_module("diffusers"), index["scheduler"][1]
        )
        out["scheduler"] = scheduler_cls.from_pretrained(str(model_dir), subfolder="scheduler")

    @staticmethod
    def _load_vae(model_dir: Path, device: str, torch_dtype: torch.dtype, cache_to_ram: bool):
        from weellm.models.vaes.lazy_vae import LazyVAEStreamer
        return LazyVAEStreamer.from_pretrained(
            model_dir / "vae",
            device=device,
            dtype=torch_dtype,
            cache_to_ram=cache_to_ram,
        )

    @staticmethod
    def _load_text_encoders(
        model_dir: Path,
        index: dict,
        device: str,
        torch_dtype: torch.dtype,
        effective_dtype: torch.dtype,
        cache_to_ram: bool,
        out: dict,
    ) -> dict:
        """Load all text encoders, inject streamers, return streamer references for later eviction."""
        te_streamers = {}

        for key in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"]:
            if key not in index:
                continue

            hf_cls_name = index[key][1]
            if hf_cls_name not in _TE_MAP:
                # Fall back to loading the full model on device
                hf_module = importlib.import_module("transformers")
                hf_cls    = getattr(hf_module, hf_cls_name)
                out[key]  = hf_cls.from_pretrained(
                    str(model_dir), subfolder=key, torch_dtype=effective_dtype
                ).to(device)
                out[key].eval()
                continue

            module_path       = _TE_MAP[hf_cls_name]
            streamer_cls_name = (
                "CLIPTextModelStreamer" if "CLIP" in hf_cls_name
                else hf_cls_name + "Streamer"
            )
            module    = importlib.import_module(module_path)
            te_cls    = getattr(module, streamer_cls_name)
            tok_key   = key.replace("text_encoder", "tokenizer")
            local_te_path = model_dir / key

            # If the subfolder doesn't exist locally, check the external-repo
            # fallback map.  Some pipelines (e.g. HiDream) reference a model
            # that lives in a separate Hub repo rather than shipping it.
            if not local_te_path.exists() and hf_cls_name in _TE_EXTERNAL_REPO:
                external_repo = _TE_EXTERNAL_REPO[hf_cls_name]
                logger.info(
                    "  [WeeLLM] '%s' subfolder not found locally — downloading from "
                    "external repo '%s' ...",
                    key, external_repo,
                )
                from huggingface_hub import snapshot_download
                downloaded = snapshot_download(
                    repo_id=external_repo,
                    allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
                )
                te_path = downloaded
                # Also update tokenizer_4 path to the same downloaded repo
                # (HiDream's tokenizer_4 is also from Llama)
                if tok_key not in out or out.get(tok_key) is None:
                    try:
                        from transformers import AutoTokenizer
                        out[tok_key] = AutoTokenizer.from_pretrained(downloaded)
                        logger.info("  [WeeLLM] Loaded %s from '%s'.", tok_key, external_repo)
                    except Exception as e:
                        logger.warning("  [WeeLLM] Could not load %s from '%s': %s", tok_key, external_repo, e)
            else:
                te_path = str(local_te_path)

            if "Qwen" in hf_cls_name or "Mistral" in hf_cls_name or "Llama" in hf_cls_name:
                if hasattr(te_cls, "from_pretrained"):
                    streamer = te_cls.from_pretrained(
                        model_dir=te_path,
                        tokenizer=out.get(tok_key),
                        device=device,
                        dtype=torch_dtype,
                        cache_to_ram=cache_to_ram,
                    )
                    if hasattr(streamer, "_ensure_initialized"):
                        streamer._ensure_initialized()
                else:
                    streamer = te_cls(
                        text_encoder_dir=te_path,
                        tokenizer_dir=str(model_dir / tok_key),
                        device=device,
                        dtype=torch_dtype,
                        cache_to_ram=cache_to_ram,
                    )
                    if hasattr(streamer, "_ensure_initialized"):
                        streamer._ensure_initialized()
            elif "CLIP" in hf_cls_name:
                hf_module = importlib.import_module("transformers")
                hf_cls    = getattr(hf_module, hf_cls_name)
                streamer  = te_cls.from_pretrained(
                    hf_cls, str(model_dir), key,
                    device=device, dtype=torch_dtype,
                    output_hidden_states=True,
                    cache_to_ram=cache_to_ram,
                )
            else:
                streamer = te_cls.from_pretrained(
                    model_dir=te_path, device=device, dtype=torch_dtype, cache_to_ram=cache_to_ram
                )
                if hasattr(streamer, "_ensure_initialized"):
                    streamer._ensure_initialized()

            te_streamers[key] = streamer
            te_model = getattr(streamer, "model", getattr(streamer, "_model", streamer))
            te_model = WeePipeline._patch_to(te_model)
            out[key] = te_model

        return te_streamers

    @staticmethod
    def _load_transformer(
        model_dir: Path,
        index: dict,
        device: str,
        torch_dtype: torch.dtype,
        prefetch: bool,
        cache_to_ram: bool,
    ):
        transformer_key = "transformer" if "transformer" in index else "unet"
        transformer_class_name = index[transformer_key][1]

        module_path = _TR_MAP.get(transformer_class_name, "")
        if not module_path:
            raise ValueError(f"Unsupported architecture: {transformer_class_name}")

        module                   = importlib.import_module(module_path)
        transformer_cls_streamer = getattr(module, transformer_class_name + "Streamer")

        if transformer_key == "unet":
            streamer = transformer_cls_streamer.from_pretrained(
                str(model_dir), device, torch_dtype, prefetch, cache_to_ram=cache_to_ram
            )
        else:
            streamer = transformer_cls_streamer.from_pretrained(
                model_dir / transformer_key,
                device=device, dtype=torch_dtype,
                prefetch=prefetch, cache_to_ram=cache_to_ram,
            )

        if hasattr(streamer, "_ensure_initialized"):
            streamer._ensure_initialized()

        return transformer_key, streamer

    @staticmethod
    def _patch_to(model):
        """
        Monkey-patch ``.to()`` so it skips meta-device tensors.
        This prevents diffusers / accelerate from crashing when they call
        ``.to(device)`` on a model that still has meta-parameter placeholders.
        """
        if not hasattr(model, "to"):
            return model

        original_to = model.to

        def safe_to(*args, **kwargs):
            def safe_convert(t):
                if t.device.type != "meta":
                    return t.to(*args, **kwargs)
                return t
            return model._apply(safe_convert)

        model.to = safe_to
        return model

    @staticmethod
    def _patch_execution_device(pipeline, device: str, te_streamers: dict) -> None:
        """Force pipeline.device / _execution_device to the real GPU when TE has meta params."""
        if not (hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None):
            return

        te = pipeline.text_encoder
        logger.debug("Pipeline text_encoder: %s", te.__class__.__name__)

        try:
            meta_params = sum(
                1 for p in te.parameters()
                if getattr(p, "device", None) is not None and p.device.type == "meta"
            )
        except Exception as exc:
            logger.debug("Unable to summarize text_encoder tensors: %s", exc)
            meta_params = 0

        if meta_params > 0:
            real_device  = torch.device(device)
            pipeline_cls = pipeline.__class__

            if not hasattr(pipeline_cls, "_weellm_original_execution_device"):
                pipeline_cls._weellm_original_execution_device = pipeline_cls._execution_device
                pipeline_cls._weellm_original_device           = pipeline_cls.device

                pipeline_cls._execution_device = property(lambda self_obj: real_device)
                pipeline_cls.device            = property(lambda self_obj: real_device)
                logger.debug(
                    "Forced pipeline execution device to %s (text_encoder has meta weights).",
                    real_device,
                )

    @staticmethod
    def _patch_scheduler(pipeline, device: str) -> None:
        """Move scheduler tensors to the correct device and patch set_timesteps/step."""
        if not hasattr(pipeline, "scheduler"):
            return

        def _contains_tensor(value):
            if torch.is_tensor(value):
                return True
            if isinstance(value, (list, tuple)):
                return any(_contains_tensor(i) for i in value)
            if isinstance(value, dict):
                return any(_contains_tensor(i) for i in value.values())
            return False

        def _move_value(value, target):
            if torch.is_tensor(value):
                return value.to(target) if value.device.type != target else value
            if isinstance(value, list):
                return [_move_value(i, target) for i in value]
            if isinstance(value, tuple):
                return tuple(_move_value(i, target) for i in value)
            if isinstance(value, dict):
                return {k: _move_value(v, target) for k, v in value.items()}
            return value

        def _move_scheduler(scheduler_obj, target_device):
            for attr_name, attr_value in list(vars(scheduler_obj).items()):
                if attr_name == "config":
                    continue
                if _contains_tensor(attr_value):
                    setattr(scheduler_obj, attr_name, _move_value(attr_value, target_device))

        _move_scheduler(pipeline.scheduler, device)

        if hasattr(pipeline.scheduler, "set_timesteps"):
            original_set_timesteps = pipeline.scheduler.set_timesteps
            original_step          = pipeline.scheduler.step

            def safe_set_timesteps(num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
                # Build kwargs only for args the scheduler actually accepts.
                import inspect
                sig    = inspect.signature(original_set_timesteps)
                params = sig.parameters
                kw: dict = {}
                if "num_inference_steps" in params:
                    kw["num_inference_steps"] = num_inference_steps
                if "device" in params:
                    kw["device"] = device
                if "sigmas" in params and sigmas is not None:
                    kw["sigmas"] = sigmas
                if "mu" in params and mu is not None:
                    kw["mu"] = mu
                if "timesteps" in params and timesteps is not None:
                    kw["timesteps"] = timesteps
                result = original_set_timesteps(**kw)
                _move_scheduler(pipeline.scheduler, device)
                return result

            def safe_step(*args, **kwargs):
                args = list(args)
                tensor_device = None
                for value in args[:3]:
                    if torch.is_tensor(value):
                        tensor_device = value.device.type
                        break
                if tensor_device is None:
                    for k in ("model_output", "sample", "timestep"):
                        value = kwargs.get(k)
                        if torch.is_tensor(value):
                            tensor_device = value.device.type
                            break
                if tensor_device is None:
                    tensor_device = device

                _move_scheduler(pipeline.scheduler, tensor_device)
                for i, value in enumerate(args[:3]):
                    if torch.is_tensor(value) and value.device.type != tensor_device:
                        args[i] = value.to(tensor_device)
                for k, value in list(kwargs.items()):
                    if torch.is_tensor(value) and value.device.type != tensor_device:
                        kwargs[k] = value.to(tensor_device)

                result = original_step(*args, **kwargs)
                _move_scheduler(pipeline.scheduler, tensor_device)
                return result

            # Assign as plain functions (not bound methods) — the originals are
            # already bound to the scheduler object, so wrapping them this way
            # avoids the self_obj first-arg inconsistency.
            pipeline.scheduler.set_timesteps = safe_set_timesteps
            pipeline.scheduler.step          = safe_step

    @staticmethod
    def _patch_pipeline_to(pipeline) -> None:
        """Patch pipeline.to() to skip modules that contain meta tensors."""
        original_pipeline_to = pipeline.to

        def safe_pipeline_to(self_obj, *args, **kwargs):
            hidden_meta_modules = {}
            for name, module in list(self_obj.components.items()):
                if isinstance(module, torch.nn.Module):
                    has_meta = (
                        any(p.device.type == "meta" for p in getattr(module, "parameters", lambda: [])())
                        or any(b.device.type == "meta" for b in getattr(module, "buffers", lambda: [])())
                    )
                    if has_meta:
                        hidden_meta_modules[name] = module
                        setattr(self_obj, name, None)
            try:
                res = original_pipeline_to(*args, **kwargs)
            finally:
                for name, module in hidden_meta_modules.items():
                    setattr(self_obj, name, module)
            return res

        pipeline.to = types.MethodType(safe_pipeline_to, pipeline)

    @staticmethod
    def _apply_optimizations(
        pipeline,
        device: str,
        cache_to_ram: bool,
        te_streamers: dict,
        transformer_key: str,
        vae_tile_size: int,
    ) -> None:
        """Apply xformers, VAE tiling, text-encoder eviction hooks, and VRAM defrag."""

        logger.info("\n[5/5] Applying Aggressive RAM Eviction...")

        cuda_available = torch.cuda.is_available() and device != "cpu"

        # xformers for pre-Ampere GPUs
        if cuda_available and torch.cuda.get_device_capability()[0] < 8:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info(
                    "      -> [WeeLLM] Enabled xFormers memory-efficient attention for older GPU (Compute < 8.0)."
                )
            except Exception:
                pass

        # VAE tiling
        try:
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
                pipeline.vae.tile_sample_min_size = vae_tile_size
                if hasattr(pipeline.vae.config, "block_out_channels"):
                    pipeline.vae.tile_latent_min_size = int(
                        vae_tile_size / (2 ** (len(pipeline.vae.config.block_out_channels) - 1))
                    )
                logger.info(
                    "      -> [WeeLLM] Enabled Aggressive VAE Tiling (tile_size=%d) to prevent decoding VRAM spikes.",
                    vae_tile_size,
                )
            elif hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
                logger.info("      -> [WeeLLM] Enabled VAE Tiling (via pipeline) to prevent decoding VRAM spikes.")
        except Exception:
            pass

        # Aggressive one-shot RAM deletion for cache_to_ram mode
        if cache_to_ram and hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
            original_vae_decode = pipeline.vae.decode

            def aggressive_vae_decode(self_obj, *args, **kwargs):
                logger.info("\n[WeeLLM] One-Shot: Freeing Transformer from RAM before VAE Decode...")
                for tr_name in ["transformer", "unet"]:
                    if hasattr(pipeline, tr_name):
                        setattr(pipeline, tr_name, None)
                gc.collect()

                # Cast float inputs (latents) to VAE's expected dtype
                tgt_dtype = getattr(self_obj, "dtype", None)
                if tgt_dtype is not None:
                    args = tuple(a.to(tgt_dtype) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
                    kwargs = {k: (v.to(tgt_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}

                return original_vae_decode(*args, **kwargs)

            pipeline.vae.decode = types.MethodType(aggressive_vae_decode, pipeline.vae)

        def _evict_module_to_meta(module, label: str) -> None:
            if module is None or not isinstance(module, torch.nn.Module):
                return
            evicted = evict_module(module)
            if cuda_available:
                torch.cuda.empty_cache()
            logger.debug("[WeeLLM Offload] Evicted %d tensors to meta (%s).", evicted, label)

        # TE eviction hook: runs on FIRST forward of transformer/unet
        def _evict_te_before_unet(module, args):
            if getattr(module, "_weellm_te_evicted", False):
                return
            module._weellm_te_evicted = True

            report_memory("Before Text Encoder Offload")
            for te_key in te_streamers.keys():
                te_mod = getattr(pipeline, te_key, None)
                _evict_module_to_meta(te_mod, te_key)

            if cache_to_ram:
                logger.info("\n[WeeLLM] One-Shot: Freeing Text Encoders from RAM...")
                for te_name in ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4",
                                 "tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"]:
                    if hasattr(pipeline, te_name):
                        setattr(pipeline, te_name, None)

            gc.collect()
            if cuda_available:
                torch.cuda.empty_cache()
            report_memory("After Text Encoder Offload")

        tr_module = getattr(pipeline, "unet", None) or getattr(pipeline, "transformer", None)
        if tr_module is not None:
            tr_module.register_forward_pre_hook(_evict_te_before_unet)

        # VRAM defrag: evict transformer before lazy VAE decode
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
            original_vae_decode_vram = pipeline.vae.decode

            def defrag_vae_decode(self_obj, *args, **kwargs):
                report_memory("Before VAE Decode (Before GC)")
                gc.collect()
                if cuda_available:
                    torch.cuda.empty_cache()
                report_memory("Before VAE Decode (After GC)")
                _evict_module_to_meta(getattr(pipeline, "transformer", None), "transformer")
                report_memory("Before VAE Decode (After Transformer Offload)")

                # Cast float inputs (latents) to VAE's expected dtype
                tgt_dtype = getattr(self_obj, "dtype", None)
                if tgt_dtype is not None:
                    args = tuple(a.to(tgt_dtype) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
                    kwargs = {k: (v.to(tgt_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}

                res = original_vae_decode_vram(*args, **kwargs)
                report_memory("After VAE Decode (Before GC)")
                gc.collect()
                if cuda_available:
                    torch.cuda.empty_cache()
                report_memory("After VAE Decode (After GC)")
                return res

            pipeline.vae.decode = types.MethodType(defrag_vae_decode, pipeline.vae)
