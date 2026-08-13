"""
qwen3_for_causal_lm.py -- Hook-based layer-streaming for Qwen3ForCausalLM (Flux2 text encoder).

Uses BaseLazyDecoderStreamer. Architecture-specific details:
  - Layer prefix: ``model.layers.{i}.``
  - Resident: ``model.embed_tokens.*``, ``model.norm.*``
  - Captures: penultimate layer only (layer 34 for Qwen3's 36 layers)
  - Special: Qwen3 rotary uses compute_default_rope_parameters
  - Forward masking: applies attention_mask to filter padding tokens (native ZImage contract)
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
    
    CRITICAL: When the model is placed into the diffusers pipeline, it will be called
    directly as text_encoder(input_ids, attention_mask, output_hidden_states=True).
    This wrapper overrides the forward method to match the native ZImage contract:
    - Returns hidden_states[-2] (penultimate layer)
    - Masks by attention_mask to remove padding
    - Returns as a modified output that diffusers can consume
    """

    def __init__(
        self,
        text_encoder_dir,
        tokenizer_dir,
        extract_layers: Optional[Tuple[int, ...]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 512,
    ):
        super().__init__(text_encoder_dir, tokenizer_dir, device, dtype, cache_to_ram, max_length)
        # Default: use penultimate layer only (matching native diffusers contract)
        self._extract_layers = extract_layers if extract_layers is not None else None

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
        # Default to penultimate layer if not explicitly set (matching native diffusers)
        if self._extract_layers is None:
            self._extract_layers = (self._num_layers - 2,)
        with default_dtype(self.dtype), init_empty_weights():
            self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_resident_extra(self) -> None:
        """Fix up Qwen3 rotary embeddings and wrap forward method for diffusers compatibility."""
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
        
        # Wrap the model's forward method to apply masking when called by diffusers
        self._install_forward_wrapper()

    def _capture_layer_indices(self) -> set:
        return set(self._extract_layers)

    def _install_forward_wrapper(self) -> None:
        """
        Wrap the model's forward method to ensure output_hidden_states=True.
        
        Diffusers expects: text_encoder(input_ids, attention_mask, output_hidden_states=True)
        The wrapper ensures we always return hidden_states even if not explicitly requested.
        """
        original_forward = self._model.forward
        
        @torch.no_grad()
        def ensure_hidden_states_forward(input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            # Call the original forward with output_hidden_states=True to get all layers
            output = original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # Always compute hidden states
                **kwargs
            )
            # Return the unmodified output. Diffusers pipeline will:
            # 1. Extract .hidden_states[-2] (the penultimate layer)
            # 2. Then mask it by attention_mask: prompt_embeds[i][prompt_masks[i]]
            return output
        
        self._model.forward = ensure_hidden_states_forward

    # -- Encoding --

    @torch.no_grad()
    def encode(self, prompt: str) -> list[torch.Tensor]:
        """
        Encode a prompt to embeddings, matching native ZImagePipeline._encode_prompt contract.
        
        Returns:
            list[torch.Tensor]: One tensor per batch item, with padding removed via attention_mask.
        """
        self._ensure_initialized()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device).bool()

        self._captured.clear()
        _ = self._model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Extract penultimate (or specified) layer
        target_layer = list(self._extract_layers)[0]  # Use first layer in extract set
        hidden = self._captured[target_layer].to(dtype=self.dtype)
        
        # Mask by valid tokens (remove padding) - matching native contract
        embeddings_list = []
        for i in range(hidden.shape[0]):
            embeddings_list.append(hidden[i][attention_mask[i]])
        
        self._captured.clear()
        clean_memory(self.device)
        return embeddings_list

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Encode token IDs to embeddings, matching native ZImagePipeline._encode_prompt contract.
        
        Returns:
            list[torch.Tensor]: One tensor per batch item, with padding removed via attention_mask.
        """
        self._ensure_initialized()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device).bool()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        self._captured.clear()
        _ = self._model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Extract penultimate (or specified) layer
        target_layer = list(self._extract_layers)[0]  # Use first layer in extract set
        hidden = self._captured[target_layer].to(dtype=self.dtype)
        
        # Mask by valid tokens (remove padding) - matching native contract
        embeddings_list = []
        for i in range(hidden.shape[0]):
            embeddings_list.append(hidden[i][attention_mask[i]])
        
        self._captured.clear()
        clean_memory(self.device)
        return embeddings_list

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