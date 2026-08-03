"""
qwen2_5_vl_streamer.py -- Streaming Qwen2.5-VL text encoder.

The text encoder is Qwen2_5_VLForConditionalGeneration (~16.6 GB, 28 layers).
We use the same hook-based strategy: resident weights stay on GPU, 
individual transformer layers stream on demand.

The encode logic applies the Qwen-Image text prompt template,
runs Qwen2.5-VL forward with output_hidden_states=True, extracts 
hidden_states[-1], masks to only actual tokens, and drops the template prefix.
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

from weellm.utils import clean_memory, report_memory
from weellm.live_seek import SafetensorsLiveSeeker


# The same prompt template used by QwenImagePipeline
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# Tokens to drop from the front (the system/user template prefix before actual answer)
PROMPT_TEMPLATE_DROP_IDX = 34


def _get_layer_keys(seeker: SafetensorsLiveSeeker, prefix: str) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix + ".")]


def _get_resident_keys(seeker: SafetensorsLiveSeeker) -> List[str]:
    """Everything except the transformer layer blocks."""
    return [k for k in seeker.weight_map.keys() if not k.startswith("model.layers.")]


def map_qwen_key(k: str) -> str:
    """Map safetensors keys to Qwen2_5_VLForConditionalGeneration module names."""
    if k.startswith("visual."):
        return "model." + k
    elif k.startswith("model."):
        return k.replace("model.", "model.language_model.", 1)
    return k


def _apply_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
    for name, tensor in state_dict.items():
        mapped_name = map_qwen_key(name)
        if tensor.is_floating_point():
            set_module_tensor_to_device(model, mapped_name, device, value=tensor, dtype=dtype)
        else:
            set_module_tensor_to_device(model, mapped_name, device, value=tensor)


def _evict_params(model: nn.Module, param_names: List[str]):
    for name in param_names:
        mapped_name = map_qwen_key(name)
        set_module_tensor_to_device(model, mapped_name, "meta")


class Qwen2_5_VLForConditionalGenerationStreamer:
    """
    Wraps Qwen2_5_VLForConditionalGeneration for streaming layer-by-layer inference.
    Only the 28 transformer layers are streamed; everything else stays resident.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker: SafetensorsLiveSeeker,
        layer_count: int,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        max_length: int = 512,
    ):
        self.model = model
        self.seeker = seeker
        self.layer_count = layer_count
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch
        self.max_length = max_length

        self._shard_order = [f"model.layers.{i}" for i in range(layer_count)]
        self._shard_name_to_pos = {s: idx for idx, s in enumerate(self._shard_order)}

        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_future_name: Optional[str] = None
        self._lock = threading.Lock()

        self._install_hooks()

    def _install_hooks(self):
        for i, shard_name in enumerate(self._shard_order):
            layer = self.model.model.language_model.layers[i]
            layer._qwen_te_shard = shard_name
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)

    def _pre_hook(self, module: nn.Module, args):
        shard_name: str = module._qwen_te_shard
        pos = self._shard_name_to_pos[shard_name]
        layer_keys = _get_layer_keys(self.seeker, shard_name)

        with self._lock:
            if self.prefetch and self._next_future_name == shard_name and self._next_future is not None:
                sd = self._next_future.result()
                self._next_future = None
                self._next_future_name = None
            else:
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)

        _apply_state_dict(self.model, sd, self.device, self.dtype)
        module._qwen_te_loaded = list(sd.keys())

        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < len(self._shard_order):
            next_name = self._shard_order[next_pos]
            next_keys = _get_layer_keys(self.seeker, next_name)
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

    def _post_hook(self, module: nn.Module, args, output):
        _evict_params(self.model, getattr(module, "_qwen_te_loaded", []))
        module._qwen_te_loaded = []
        return output

    @torch.no_grad()
    def encode(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          prompt_embeds:      (1, seq_len, 3584)
          prompt_embeds_mask: (1, seq_len) long tensor
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        drop_idx = PROMPT_TEMPLATE_DROP_IDX
        txt = [PROMPT_TEMPLATE.format(p) for p in prompt]

        tokens = self.tokenizer(
            txt,
            max_length=self.max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # (B, seq, hidden)

        # Extract non-padded tokens for each item then re-pad to same length
        split_hidden = []
        attn = tokens.attention_mask
        for b in range(hidden.shape[0]):
            mask_b = attn[b].bool()
            split_hidden.append(hidden[b][mask_b])

        split_hidden = [e[drop_idx:] for e in split_hidden]
        attn_masks = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden]
        max_len = max(e.size(0) for e in split_hidden)

        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))])
            for u in split_hidden
        ]).to(dtype=self.dtype)

        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0))])
            for u in attn_masks
        ])

        # Truncate to max_length
        prompt_embeds = prompt_embeds[:, :self.max_length]
        encoder_attention_mask = encoder_attention_mask[:, :self.max_length]

        return prompt_embeds, encoder_attention_mask

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        max_length: int = 512,
    ) -> "Qwen2_5_VLForConditionalGenerationStreamer":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_dir = Path(model_dir)

        print("  [TE 1/3] Initializing LiveSeeker on Qwen text encoder weights ...")
        seeker = SafetensorsLiveSeeker(model_dir)
        print(f"    Found {len(seeker.weight_map)} tensors.")

        print("  [TE 2/3] Instantiating Qwen2_5_VLForConditionalGeneration on meta device ...")
        with init_empty_weights():
            from transformers import Qwen2_5_VLConfig
            cfg = Qwen2_5_VLConfig.from_pretrained(str(model_dir))
            model = Qwen2_5_VLForConditionalGeneration(cfg)
        model.eval()

        # Move non-meta buffers to device
        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type != "meta":
                set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("  [TE 3/3] Loading resident Qwen text encoder tensors to GPU ...")
        resident_keys = _get_resident_keys(seeker)
        resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        _apply_state_dict(model, resident_sd, device, dtype)
        del resident_sd
        clean_memory(device)

        layer_count = len(model.model.language_model.layers)
        print(f"    -> {layer_count} Qwen layers will stream on-demand.")
        report_memory("After Qwen text encoder init")

        return cls(
            model=model,
            seeker=seeker,
            layer_count=layer_count,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
            max_length=max_length,
        )
