"""
glm_model.py -- Hook-based layer-streaming for the GlmModel (CogView4 text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``layers.{i}.``
  - Resident: ``embed_tokens.*``, ``norm.*``
  - Captures: penultimate layer (layer N-2, to match diffusers hidden_states[-2])
  - Special: rotary_emb buffers loaded to GPU as float32
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from weellm.utils import default_dtype
from weellm.memory import place_tensors
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer



class GlmModelStreamer(BaseLazyDecoderStreamer):
    """Streaming text encoder for GlmModel (CogView4)."""

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        extract_layers: Tuple[int, ...] = (39,),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 1024,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)
        # Will be resolved to (num_layers - 2,) inside _load_model_skeleton
        self._extract_layers = extract_layers

    # -- Abstract implementations --

    @property
    def _model_name(self) -> str:
        return "GLM"

    def _layer_prefix(self, idx: int) -> str:
        return f"layers.{idx}."

    def _resident_key_filter(self, key: str) -> bool:
        return key.startswith("embed_tokens.") or key.startswith("norm.")

    def _get_model_layers(self) -> nn.ModuleList:
        return self._model.layers

    def _load_model_skeleton(self) -> None:
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_hidden_layers
        # Match diffusers: hidden_states[-2] → capture penultimate layer
        self._extract_layers = (self._num_layers - 2,)
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_resident_extra(self) -> None:
        """Move rotary_emb buffers to GPU as float32."""
        rotary = self._model.rotary_emb
        for buf_name, buf in list(rotary.named_buffers()):
            if buf.device.type != self.device:
                place_tensors(self._model, {f"rotary_emb.{buf_name}": buf.float()}, self.device, torch.float32)

    def _capture_layer_indices(self) -> set:
        return set(self._extract_layers)

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_initialized()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(self.device)

        # Pad to multiple of 16 (required by GlmModel attention)
        pad_len = (16 - (input_ids.shape[1] % 16)) % 16
        if pad_len > 0:
            pad_ids = torch.full(
                (input_ids.shape[0], pad_len),
                fill_value=self._tokenizer.pad_token_id,
                dtype=input_ids.dtype, device=input_ids.device,
            )
            input_ids = torch.cat([pad_ids, input_ids], dim=1)

        self._captured.clear()
        _ = self._model(input_ids=input_ids, use_cache=False)

        stacked = torch.stack([self._captured[k] for k in self._extract_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        self._captured.clear()
        return prompt_embeds

    @torch.no_grad()
    def encode_ids(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        self._ensure_initialized()
        input_ids = input_ids.to(self.device)
        pad_len = (16 - (input_ids.shape[1] % 16)) % 16
        if pad_len > 0:
            pad_ids = torch.full(
                (input_ids.shape[0], pad_len),
                fill_value=self._tokenizer.pad_token_id,
                dtype=input_ids.dtype, device=input_ids.device,
            )
            input_ids = torch.cat([pad_ids, input_ids], dim=1)
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
        max_length: int = 1024,
    ) -> "GlmModelStreamer":
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
