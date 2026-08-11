"""
llama_for_causal_lm.py -- Hook-based layer-streaming for LlamaForCausalLM (HiDream text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``model.layers.{i}.``
  - Resident: ``model.embed_tokens.*``, ``model.norm.*``
  - Captures: ALL layers (HiDream stacks every hidden state)
  - Special: rotary_emb buffers loaded as float32
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from weellm.utils import default_dtype, clean_memory
from weellm.memory import place_tensors, pin_module_to_cpu
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer


class LlamaForCausalLMStreamer(BaseLazyDecoderStreamer):
    """
    Streaming text encoder for Llama 3 (used in HiDream).
    Captures outputs from ALL decoder layers because HiDream stacks all hidden states.
    """

    # -- Abstract implementations --

    @property
    def _model_name(self) -> str:
        return "Llama"

    def _layer_prefix(self, idx: int) -> str:
        return f"model.layers.{idx}."

    def _resident_key_filter(self, key: str) -> bool:
        return "norm" in key

    def _cpu_resident_key_filter(self, key: str) -> bool:
        return "embed_tokens" in key

    def _get_model_layers(self) -> nn.ModuleList:
        return self._model.model.layers

    def _load_model_skeleton(self) -> None:
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_hidden_layers
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_resident_extra(self) -> None:
        """Move rotary_emb buffers to GPU."""
        rotary = self._model.model.rotary_emb
        for buf_name, buf in list(rotary.named_buffers()):
            if buf.device.type != self.device:
                place_tensors(self._model, {f"model.rotary_emb.{buf_name}": buf.float()}, self.device, torch.float32)

        # Embeddings are on CPU, run them on CPU!
        pin_module_to_cpu(self._model, "model.embed_tokens")

    def _capture_layer_indices(self) -> set:
        # HiDream needs every layer
        return set(range(self._num_layers))

    # -- Encoding --

    def encode(self, prompt) -> torch.Tensor:
        raise NotImplementedError("Use encode_ids() for HiDream Llama")

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
        _ = self._model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # HiDream: stack all layer outputs (layers[0]..layers[N-1])
        stacked = torch.stack([self._captured[k] for k in range(self._num_layers)], dim=0)
        self._captured.clear()
        clean_memory(self.device)
        return stacked

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        tokenizer=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
        max_length: int = 128,
    ) -> "LlamaForCausalLMStreamer":
        model_dir = Path(model_dir)
        tokenizer_dir = model_dir.parent / "tokenizer_4" if model_dir.name == "text_encoder_4" else model_dir
        return cls(
            text_encoder_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            max_length=max_length,
        )
