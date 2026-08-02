"""
text_encoder_streamer.py -- Hook-based layer-streaming for the Qwen3 text encoder.

Uses the Live Seek Architecture to load weights directly from Hugging Face shards
without creating any duplicated files on disk.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weellm.core.utils import clean_memory
from weellm.core.live_seek import SafetensorsLiveSeeker


def _get_layer_keys(seeker: SafetensorsLiveSeeker, layer_idx: int) -> List[str]:
    prefix = f"model.layers.{layer_idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]

def _get_resident_keys(seeker: SafetensorsLiveSeeker) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith("model.embed_tokens.") or k.startswith("model.norm.")]


class StreamingQwen3TextEncoder:
    """
    Hook-based streaming text encoder (Qwen3).
    Extracts output from target decoder layers (e.g., layers 14, 21, 35) directly
    via forward hooks.
    """

    def __init__(
        self,
        text_encoder_dir: Path | str,
        tokenizer_dir: Path | str,
        extract_layers: Tuple[int, ...] = (14, 21, 35),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.extract_layers = extract_layers
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self._seeker: Optional[SafetensorsLiveSeeker] = None
        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._num_layers: int = 0
        self._initialized = False

        self._captured: Dict[int, torch.Tensor] = {}
        self._hook_handles = []

        self._executor: Optional[ThreadPoolExecutor] = None
        self._next_future = None
        self._next_future_idx: Optional[int] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return
        print("Initialising streaming Qwen3 text encoder ...")
        self._seeker = SafetensorsLiveSeeker(self.text_encoder_dir)
        self._load_model_skeleton()
        self._load_tokenizer()
        self._load_resident_modules()
        self._install_hooks()
        
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="te_gpu_load"
        )
        self._initialized = True
        print("Text encoder ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_hidden_layers
        with init_empty_weights():
            self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), trust_remote_code=True
        )

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        self._place_tensors(resident_sd)
        del resident_sd

        rotary = self._model.model.rotary_emb
        for buf_name, buf in list(rotary.named_buffers()):
            if buf.device.type != self.device:
                set_module_tensor_to_device(
                    self._model, f"model.rotary_emb.{buf_name}",
                    self.device, value=buf.float()
                )

        clean_memory(self.device)

    def _place_tensors(self, state_dict: Dict[str, torch.Tensor]):
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self._model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self._model, name, self.device, value=tensor)

    def _evict_layer(self, state_dict: Dict[str, torch.Tensor]):
        for name in state_dict.keys():
            set_module_tensor_to_device(self._model, name, "meta")

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self):
        for layer_idx in range(self._num_layers):
            layer = self._model.model.layers[layer_idx]
            layer._te_layer_idx = layer_idx

            h_pre = layer.register_forward_pre_hook(self._layer_pre_hook)
            h_post = layer.register_forward_hook(self._layer_post_hook)
            self._hook_handles.extend([h_pre, h_post])

            if layer_idx in self.extract_layers:
                h_cap = layer.register_forward_hook(self._capture_hook)
                self._hook_handles.append(h_cap)

    def _layer_pre_hook(self, module: nn.Module, args):
        idx = module._te_layer_idx
        layer_keys = _get_layer_keys(self._seeker, idx)

        with self._lock:
            if self._next_future_idx == idx and self._next_future is not None:
                gpu_sd = self._next_future.result()
                self._next_future = None
                self._next_future_idx = None
            else:
                gpu_sd = self._seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        self._place_tensors(gpu_sd)
        module._te_loaded_sd = gpu_sd

        next_idx = idx + 1
        if next_idx < self._num_layers and self._executor is not None:
            next_layer_keys = _get_layer_keys(self._seeker, next_idx)
            with self._lock:
                self._next_future = self._executor.submit(
                    self._seeker.get_tensors, next_layer_keys, self.device, self.dtype
                )
                self._next_future_idx = next_idx

    def _layer_post_hook(self, module: nn.Module, args, output):
        self._evict_layer(getattr(module, "_te_loaded_sd", {}))
        module._te_loaded_sd = {}
        return output

    def _capture_hook(self, module: nn.Module, args, output):
        idx = module._te_layer_idx
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured[idx] = hidden.detach().clone()
        return output

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        self._ensure_initialized()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self._captured.clear()

        _ = self._model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )

        stacked = torch.stack([self._captured[k] for k in self.extract_layers], dim=1)
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

        _ = self._model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )

        stacked = torch.stack([self._captured[k] for k in self.extract_layers], dim=1)
        B, num_captured, seq, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq, num_captured * hidden)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        self._captured.clear()
        clean_memory(self.device)
        return prompt_embeds

    @property
    def tokenizer(self):
        self._ensure_initialized()
        return self._tokenizer
