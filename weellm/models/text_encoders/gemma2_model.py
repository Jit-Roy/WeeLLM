"""
gemma2_model.py -- Hook-based layer-streaming for the Gemma2Model (Lumina2 text encoder).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModel, AutoTokenizer

from weellm.utils import clean_memory
from weellm.seeker import get_seeker


def _get_layer_keys(seeker, layer_idx: int) -> List[str]:
    prefix = f"layers.{layer_idx}."
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix)]

def _get_resident_keys(seeker) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith("embed_tokens.") or k.startswith("norm.")]


class Gemma2ModelStreamer:
    """
    Hook-based streaming text encoder (Gemma2Model).
    Extracts output from the penultimate hidden state (the last transformer layer before final norm)
    directly via forward hooks.
    """

    def __init__(
        self,
        text_encoder_dir: Path | str,
        tokenizer_dir: Path | str,
        extract_layers: Tuple[int, ...] = (-2,),  # Will be resolved to num_layers - 2
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        max_length: int = 256,
    ):
        self.text_encoder_dir = Path(text_encoder_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.extract_layers = extract_layers
        self.device = device
        self.dtype = dtype
        self.cache_to_ram = cache_to_ram
        self.max_length = max_length

        self._seeker: Optional[Any] = None
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

    def _ensure_initialized(self):
        if self._initialized:
            return
        print("Initialising streaming Gemma2 text encoder ...")
        self._seeker = get_seeker(self.text_encoder_dir, cache_to_ram=self.cache_to_ram)
        self._load_model_skeleton()
        self._load_tokenizer()
        self._load_resident_modules()
        self._install_hooks()
        
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemma_gpu_load"
        )
        self._initialized = True
        print("Gemma2 Text encoder ready (streaming via Live Seek).")

    def _load_model_skeleton(self):
        config = AutoConfig.from_pretrained(str(self.text_encoder_dir), trust_remote_code=True)
        self._num_layers = config.num_hidden_layers
        
        # Resolve negative layer indices
        self.extract_layers = tuple(
            self._num_layers + l if l < 0 else l for l in self.extract_layers
        )
        
        with init_empty_weights():
            self._model = AutoModel.from_config(config, trust_remote_code=True)
        self._model.eval()

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), trust_remote_code=False
        )
        if self._tokenizer.padding_side is None:
            self._tokenizer.padding_side = "right"

    def _load_resident_modules(self):
        resident_keys = _get_resident_keys(self._seeker)
        resident_sd = self._seeker.get_tensors(resident_keys, device="cpu", dtype=self.dtype)
        self._place_tensors(resident_sd)
        del resident_sd

        for buf_name, buf in list(self._model.named_buffers()):
            if buf.device.type != self.device:
                set_module_tensor_to_device(
                    self._model, buf_name, self.device, value=buf.to(self.dtype) if buf.is_floating_point() else buf
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

    def _install_hooks(self):
        for layer_idx in range(self._num_layers):
            layer = self._model.layers[layer_idx]
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

    @torch.no_grad()
    def encode(self, prompt: str | List[str]) -> torch.Tensor:
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
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self._captured.clear()

        _ = self._model(
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

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        tokenizer=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
        max_length: int = 256
    ) -> "Gemma2ModelStreamer":
        tokenizer_dir = Path(model_dir).parent / "tokenizer" if Path(model_dir).name == "text_encoder" else model_dir
        return cls(
            text_encoder_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            max_length=max_length,
        )
