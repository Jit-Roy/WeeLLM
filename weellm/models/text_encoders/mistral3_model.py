"""
mistral3_model.py -- Streaming for Mistral3 (ERNIE-Image text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``layers.{i}.`` or ``model.layers.{i}.`` (Depends on AutoModel)
  - Resident: ``embed_tokens.*``, ``norm.*``
  - Captures: last layer hidden states
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from weellm.utils import default_dtype, clean_memory
from weellm.memory import place_tensors, pin_module_to_cpu
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer


class Mistral3ModelStreamer(BaseLazyDecoderStreamer):
    """Streaming text encoder for Mistral3Model."""

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 512,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)

    # -- Abstract implementations --

    @property
    def _model_name(self) -> str:
        return "Mistral3Model"
        
    @property
    def model(self):
        # Diffusers will call pipeline.text_encoder(input_ids). We expose the inner text model 
        # (wrapped in ModelWrapper) to bypass the multimodal Mistral3Model.forward bugs.
        self._ensure_initialized()
        return self._model.language_model

    def _layer_prefix(self, idx: int) -> str:
        return f"language_model.model.layers.{idx}."

    def _resident_key_filter(self, key: str) -> bool:
        return "norm" in key

    def _cpu_resident_key_filter(self, key: str) -> bool:
        return "embed_tokens" in key

    def _get_model_layers(self) -> nn.ModuleList:
        return self._model.language_model.model.layers

    def _load_model_skeleton(self) -> None:
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        if hasattr(config, "text_config"):
            self._num_layers = config.text_config.num_hidden_layers
        else:
            self._num_layers = config.num_hidden_layers
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
            
        # Diffusers calls self._model directly. The outer Mistral3Model inherits a get_input_embeddings
        # that raises NotImplementedError, so we unconditionally override it.
        self._model.get_input_embeddings = lambda: self._model.language_model.model.embed_tokens
            
        class ModelWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)
                
        self._model.language_model = ModelWrapper(self._model.language_model)
        self._model.eval()

    def _load_resident_extra(self) -> None:
        """Move rotary_emb buffers to GPU as float32."""
        rotary = getattr(self._model.language_model.model, "rotary_emb", None)
        prefix = "language_model.model."

        if rotary is not None:
            for buf_name, buf in list(rotary.named_buffers()):
                if buf.device.type != self.device:
                    place_tensors(self._model, {f"{prefix}rotary_emb.{buf_name}": buf.float()}, self.device, torch.float32)
                
        # Embeddings are on CPU, run them on CPU!
        pin_module_to_cpu(self._model, f"{prefix}embed_tokens")

    def _capture_layer_indices(self) -> set:
        return {self._num_layers - 1}

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_initialized()

        # Just standard tokenization for Ernie-Image
        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self._captured.clear()
        _ = self._model.language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        prompt_embeds = self._captured[self._num_layers - 1]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        self._captured.clear()
        clean_memory(self.device)
        return prompt_embeds

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_initialized()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self._captured.clear()
        _ = self._model.language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        prompt_embeds = self._captured[self._num_layers - 1]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        self._captured.clear()
        clean_memory(self.device)
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
        max_length: int = 512,
    ) -> "Mistral3ModelStreamer":
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
