"""
gemma2_model.py -- Hook-based layer-streaming for Gemma2Model (Lumina2 text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``layers.{i}.``
  - Resident: ``embed_tokens.*``, ``norm.*``
  - Captures: configurable extract_layers (defaults to penultimate)
  - Special: all buffers moved to GPU; padding_side forced to "right"
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from weellm.utils import default_dtype
from weellm.memory import place_tensors
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer


class Gemma2ModelStreamer(BaseLazyDecoderStreamer):
    """Streaming text encoder for Gemma2Model (Lumina2)."""

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        extract_layers: Tuple[int, ...] = (-2,),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 256,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)
        self._extract_layers = extract_layers  # resolved after skeleton load

    # -- Abstract implementations --

    @property
    def _model_name(self) -> str:
        return "Gemma2"

    def _layer_prefix(self, idx: int) -> str:
        return f"layers.{idx}."

    def _resident_key_filter(self, key: str) -> bool:
        return key.startswith("embed_tokens.") or key.startswith("norm.")

    def _get_model_layers(self) -> nn.ModuleList:
        return self._model.layers

    def _load_model_skeleton(self) -> None:
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_hidden_layers
        # Resolve negative indices
        self._extract_layers = tuple(
            self._num_layers + l if l < 0 else l for l in self._extract_layers
        )
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), trust_remote_code=False
        )
        if self._tokenizer.padding_side is None:
            self._tokenizer.padding_side = "right"

    def _load_resident_extra(self) -> None:
        """Move all model buffers to GPU."""
        for buf_name, buf in list(self._model.named_buffers()):
            if buf.device.type != self.device:
                place_tensors(
                    self._model, {buf_name: buf.to(self.dtype) if buf.is_floating_point() else buf},
                    self.device, self.dtype,
                )

    def _capture_layer_indices(self) -> set:
        return set(self._extract_layers)

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt) -> torch.Tensor:
        self._ensure_initialized()
        prompt = [prompt] if isinstance(prompt, str) else prompt

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self._captured.clear()
        _ = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

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
    ) -> "Gemma2ModelStreamer":
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
