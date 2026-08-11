"""
qwen3_for_causal_lm.py -- Hook-based layer-streaming for Qwen3ForCausalLM (Flux2 text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``model.layers.{i}.``
  - Resident: ``model.embed_tokens.*``, ``model.norm.*``
  - Captures: configurable multi-layer (default: layers 14, 21, 35 for Flux2-Klein)
  - Special: Qwen3 rotary uses compute_default_rope_parameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from weellm.utils import default_dtype, clean_memory
from weellm.memory import place_tensors, pin_module_to_cpu
from weellm.models.text_encoders.base_te_streamer import BaseLazyDecoderStreamer


class Qwen3ForCausalLMStreamer(BaseLazyDecoderStreamer):
    """
    Streaming text encoder for Qwen3ForCausalLM (Flux2-Klein).
    Captures multi-layer hidden states and concatenates them.
    """

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        extract_layers: Tuple[int, ...] = (14, 21, 35),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 512,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)
        self._extract_layers = extract_layers

    # -- Abstract implementations --

    @property
    def _model_name(self) -> str:
        return "Qwen3"

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
        """Fix up Qwen3 rotary embeddings (may use compute_default_rope_parameters)."""
        rotary = self._model.model.rotary_emb
        if hasattr(rotary, "compute_default_rope_parameters"):
            inv_freq, _ = rotary.compute_default_rope_parameters(self._model.config, device="cpu")
            place_tensors(self._model, {"model.rotary_emb.inv_freq": inv_freq}, self.device, torch.float32)
            if hasattr(rotary, "original_inv_freq"):
                place_tensors(self._model, {"model.rotary_emb.original_inv_freq": inv_freq.clone()}, self.device, torch.float32)
        else:
            for buf_name, buf in list(rotary.named_buffers()):
                if buf.device.type != self.device:
                    place_tensors(self._model, {f"model.rotary_emb.{buf_name}": buf.float()}, self.device, torch.float32)
                    
        # Embeddings are on CPU, run them on CPU!
        pin_module_to_cpu(self._model, "model.embed_tokens")

    def _capture_layer_indices(self) -> set:
        # hooks use 0-indexed layer idx; extract_layers are 1-indexed (HF convention)
        return {L - 1 for L in self._extract_layers}

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_initialized()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self._captured.clear()
        _ = self._model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        target_layers = [L - 1 for L in self._extract_layers]
        stacked = torch.stack([self._captured[k] for k in target_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
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
        _ = self._model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        target_layers = [L - 1 for L in self._extract_layers]
        stacked = torch.stack([self._captured[k] for k in target_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
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
    ) -> "Qwen3ForCausalLMStreamer":
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


# Aliases kept for backwards compatibility
Qwen3ModelStreamer       = Qwen3ForCausalLMStreamer
Qwen2ForCausalLMStreamer = Qwen3ForCausalLMStreamer