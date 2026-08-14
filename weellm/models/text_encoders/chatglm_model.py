"""
chatglm_model.py -- Hook-based layer-streaming for ChatGLMModel (Kolors text encoder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from weellm.utils import default_dtype
from weellm.memory import place_tensors, pin_module_to_cpu
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer


class ChatGLMModelStreamer(BaseLazyDecoderStreamer):
    """Streaming text encoder for ChatGLMModel (Kolors)."""

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        extract_layers: Tuple[int, ...] = (27,),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 256,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)
        self._extract_layers = extract_layers

    # -- Abstract implementations --

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch = False

    @property
    def _model_name(self) -> str:
        return "ChatGLM"

    def _layer_prefix(self, idx: int) -> str:
        return f"transformer.encoder.layers.{idx}."

    def _resident_key_filter(self, key: str) -> bool:
        return "norm" in key or "position_embeddings" in key

    def _cpu_resident_key_filter(self, key: str) -> bool:
        return "word_embeddings" in key

    def _get_model_layers(self) -> nn.ModuleList:
        return self._model.transformer.encoder.layers

    def _load_model_skeleton(self) -> None:
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_layers
        self._extract_layers = (self._num_layers - 2,)
        
        if not hasattr(config, "max_length"):
            config.max_length = getattr(config, "seq_length", 2048)
        if not hasattr(config, "use_cache"):
            config.use_cache = True
            
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
            
        # Patch forward to force all inputs AND outputs to device, while keeping word_embeddings on CPU
        orig_forward = self._model.forward
        _dev = self.device
        _dtype = self.dtype
        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=_dev, dtype=_dtype if x.is_floating_point() else None)
            return x
        def _move_output(out):
            """Recursively move all tensors in an output (tuple/namedtuple/tensor) to device."""
            if isinstance(out, torch.Tensor):
                return _move(out)
            if isinstance(out, dict):
                moved_dict = {k: _move_output(v) for k, v in out.items()}
                if type(out) is dict:
                    return moved_dict
                # For huggingface ModelOutput classes, try to reconstruct using kwargs
                try:
                    return type(out)(**moved_dict)
                except Exception:
                    # Fallback to in-place update if initialization fails
                    for k, v in moved_dict.items():
                        out[k] = v
                        if hasattr(out, k):
                            setattr(out, k, v)
                    return out
            if isinstance(out, (list, tuple)):
                moved = [_move_output(v) for v in out]
                try:
                    return type(out)(*moved)   # namedtuple / dataclass
                except TypeError:
                    return type(out)(moved)
            return out
        def new_forward(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            
            if input_ids is not None and inputs_embeds is None:
                # Run embedding on CPU (where word_embeddings is), the hook will move output to GPU
                kwargs['inputs_embeds'] = self._model.transformer.embedding(input_ids)
                
            new_args = tuple(a.to(_dev) if isinstance(a, torch.Tensor) else a for a in args)
            new_kwargs = {k: v.to(_dev) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            result = orig_forward(*new_args, **new_kwargs)
            # Move all output tensors back to GPU (output_layer may have run on CPU)
            return _move_output(result)
        self._model.forward = new_forward
            
        self._model.eval()

    def _load_resident_extra(self) -> None:
        """Keep rotary buffers on the active device; pin output head to CPU if needed.
        Keep embeddings on CPU to save VRAM and hook it to move output to GPU."""
        if hasattr(self._model.transformer, "rotary_pos_emb"):
            rotary = self._model.transformer.rotary_pos_emb
            if hasattr(rotary, "inv_freq"):
                rotary.inv_freq = rotary.inv_freq.to(device=self.device, dtype=self.dtype)
            rotary.to(device=self.device, dtype=self.dtype)

        if hasattr(self._model.transformer, "output_layer"):
            self._model.transformer.output_layer.to(dtype=self.dtype)
            pin_module_to_cpu(self._model, "transformer.output_layer")
            
        if hasattr(self._model.transformer, "embedding"):
            def _embedding_forward_hook(module, args, output):
                return output.to(device=self.device, dtype=self.dtype)
            self._model.transformer.embedding.register_forward_hook(_embedding_forward_hook)

    def _load_tokenizer(self) -> None:
        print("DEBUG: _load_tokenizer starting", flush=True)
        import importlib.util
        import logging
        logger = logging.getLogger("weellm")
        local_repo = Path(self.tokenizer_dir)
        model_dir = local_repo.parent if local_repo.name == "tokenizer" else local_repo
        
        tokenizer_module = None
        for candidate in [
            local_repo / "tokenization_chatglm.py",
            local_repo / "tokenizer" / "tokenization_chatglm.py",
            model_dir / "tokenizer" / "tokenization_chatglm.py",
        ]:
            if candidate.exists():
                tokenizer_module = candidate
                break

        if tokenizer_module is not None:
            try:
                module_name = f"weellm_custom_tokenizer_ChatGLM_{abs(hash(str(tokenizer_module)))}"
                spec = importlib.util.spec_from_file_location(module_name, tokenizer_module)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                tokenizer_cls = getattr(mod, "ChatGLMTokenizer", None)
                vocab_path = None
                for possible in [
                    local_repo / "tokenizer.model",
                    local_repo / "tokenizer" / "tokenizer.model",
                    model_dir / "tokenizer" / "tokenizer.model",
                ]:
                    if possible.exists():
                        vocab_path = possible
                        break
                if tokenizer_cls is not None and vocab_path is not None:
                    self._tokenizer = tokenizer_cls(vocab_file=str(vocab_path))
                    return
            except Exception as e:
                logger.warning(f"Failed to load custom ChatGLMTokenizer: {e}")
        
        super()._load_tokenizer()

    def _layer_post_hook(self, module: nn.Module, args, output):
        idx = getattr(module, "_te_layer_idx", -1)
        if idx == -1: return output

        from weellm.memory import evict_module
        evict_module(module)
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return output

    def _layer_pre_hook(self, module: nn.Module, args):
        idx = getattr(module, "_te_layer_idx", -1)
        if idx == -1: return args

        layer_keys = [k for k in self._seeker.weight_map if k.startswith(self._layer_prefix(idx))]

        with self._lock:
            if self._next_future_idx == idx and self._next_future is not None:
                gpu_sd = self._next_future.result()
                self._next_future = None
                self._next_future_idx = -1
            else:
                gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

            self._place_tensors(gpu_sd)
            gpu_sd.clear()
            import gc; gc.collect()

        if getattr(self, "prefetch", True) and (idx + 1) < self._num_layers:
            next_keys = [k for k in self._seeker.weight_map if k.startswith(self._layer_prefix(idx + 1))]
            self._next_future_idx = idx + 1
            self._next_future     = self._executor.submit(
                self._seeker.get_tensors, next_keys, self.device, self.dtype
            )
        
        return args

        return args

    def _capture_layer_indices(self) -> set:
        return set(self._extract_layers)

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_initialized()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        # Keep input_ids on CPU since word_embeddings is on CPU
        input_ids = inputs["input_ids"].to("cpu")

        self._captured.clear()
        
        _ = self._model(input_ids=input_ids, use_cache=False)

        stacked = torch.stack([self._captured[k] for k in self._extract_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        self._captured.clear()
        return prompt_embeds

    def __call__(self, *args, **kwargs):
        self._ensure_initialized()
        return super().__call__(*args, **kwargs)

    @torch.no_grad()
    def encode_ids(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        self._ensure_initialized()
        # Keep input_ids on CPU since word_embeddings is on CPU
        input_ids = input_ids.to("cpu")
        
        self._captured.clear()
        _ = self._model(input_ids=input_ids, use_cache=False)
        
        stacked = torch.stack([self._captured[k] for k in self._extract_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        
        self._captured.clear()
        return prompt_embeds

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        tokenizer=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
        max_length: int = 256,
    ) -> "ChatGLMModelStreamer":
        model_dir = Path(model_dir)
        tokenizer_dir = model_dir.parent / "tokenizer" if model_dir.name == "text_encoder" else model_dir
        return cls(
            text_encoder_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            max_length=max_length,
        )
