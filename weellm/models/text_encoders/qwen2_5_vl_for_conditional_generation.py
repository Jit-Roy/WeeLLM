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
from weellm.utils import default_dtype
from weellm.utils import clean_memory, report_memory
from weellm.memory import place_tensors, evict_module
from weellm.seeker import get_seeker
from accelerate.utils.modeling import set_module_tensor_to_device


# The same prompt template used by QwenImagePipeline
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# Tokens to drop from the front (the system/user template prefix before actual answer)
PROMPT_TEMPLATE_DROP_IDX = 34


def _get_layer_keys(seeker, prefix: str) -> List[str]:
    return [k for k in seeker.weight_map.keys() if k.startswith(prefix + ".")]


def _get_resident_keys(seeker, is_edit_model: bool = False) -> List[str]:
    """Everything except the transformer layer blocks, lm_head, and optionally the visual tower."""
    keys = []
    for k in seeker.weight_map.keys():
        if k.startswith("model.layers."):
            continue
        if k.startswith("lm_head."):
            continue
        if not is_edit_model and k.startswith("visual."):
            continue
        keys.append(k)
    return keys


def map_qwen_key(k: str) -> str:
    """Map safetensors keys to Qwen2_5_VLForConditionalGeneration module names."""
    if k.startswith("visual."):
        return "model." + k
    elif k.startswith("model."):
        return k.replace("model.", "model.language_model.", 1)
    return k


def _apply_state_dict(model, state_dict, device, dtype):
    """Remap Qwen-specific key names then place tensors onto the model."""
    mapped = {map_qwen_key(k): v for k, v in state_dict.items()}
    place_tensors(model, mapped, device, dtype)


class Qwen2_5_VLForConditionalGenerationStreamer:
    """
    Wraps Qwen2_5_VLForConditionalGeneration for streaming layer-by-layer inference.
    Only the 28 transformer layers are streamed; everything else stays resident.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        layer_count: int,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
        prefetch: bool = True,
        max_length: int = 512,
    ):
        self.model = model
        self.seeker = seeker
        self.layer_count = layer_count
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.cache_to_ram = cache_to_ram
        self.prefetch = prefetch
        self.max_length = max_length

        self._shard_order = [f"model.layers.{i}" for i in range(layer_count)]
        self._shard_name_to_pos = {s: idx for idx, s in enumerate(self._shard_order)}

        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_future_name: Optional[str] = None
        self._lock = threading.Lock()


        self._install_hooks()
        self._install_qwen_root_forward_patch()

    def _install_hooks(self):
        for i, shard_name in enumerate(self._shard_order):
            layer = self.model.model.language_model.layers[i]
            layer._qwen_te_shard = shard_name
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)

    def _install_qwen_root_forward_patch(self):
        """Patch the root forward to bypass the multimodal masking logic for pure-text prompts."""
        try:
            original_forward = self.model.forward

            def patched_qwen_forward(self_obj, *args, **kwargs):
                if kwargs.get("pixel_values") is None and kwargs.get("pixel_values_videos") is None:
                    # Pure-text bypass: route through inner language_model directly
                    input_ids = kwargs.get("input_ids")
                    if input_ids is None and len(args) > 0:
                        input_ids = args[0]
                    inner_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": kwargs.get("attention_mask"),
                        "position_ids": kwargs.get("position_ids"),
                        "past_key_values": kwargs.get("past_key_values"),
                        "inputs_embeds": kwargs.get("inputs_embeds"),
                        "use_cache": kwargs.get("use_cache", False),
                        "output_attentions": kwargs.get("output_attentions", False),
                        "output_hidden_states": True,
                        "return_dict": True,
                        "cache_position": kwargs.get("cache_position"),
                    }
                    inner_kwargs = {k: v for k, v in inner_kwargs.items() if v is not None}               
                    inner_out = self_obj.model.language_model(**inner_kwargs)
                    logits = self_obj.lm_head(inner_out[0])
                    
                    past_kv = getattr(inner_out, "past_key_values", inner_out[1] if isinstance(inner_out, tuple) and len(inner_out) > 1 else None)
                    hidden_states = getattr(inner_out, "hidden_states", inner_out[2] if isinstance(inner_out, tuple) and len(inner_out) > 2 else None)
                    
                    # Diffusers relies on hidden_states[-1]. If the inner model didn't return all hidden states,
                    # we can fallback to providing just the last hidden state in a tuple.
                    if hidden_states is None:
                        hidden_states = (inner_out[0],)
                        
                    from transformers.modeling_outputs import CausalLMOutputWithPast
                    return CausalLMOutputWithPast(
                        loss=None,
                        logits=logits,
                        past_key_values=past_kv,
                        hidden_states=hidden_states,
                        attentions=getattr(inner_out, "attentions", None),
                    )

                out = original_forward(*args, **kwargs)
                
                # Fix for missing hidden states in transformers output capturing hook when layers are streaming
                if getattr(out, "hidden_states", None) is None and kwargs.get("output_hidden_states"):
                    last_hidden = getattr(self, "_last_hidden_state", None)
                    if last_hidden is not None:
                        norm = getattr(self_obj.model.language_model, "norm", None)
                        if norm is not None:
                            last_hidden = norm(last_hidden)
                        # We only need the last hidden state for diffusers
                        out.hidden_states = (last_hidden,)
                        
                return out

            import types
            self.model.forward = types.MethodType(patched_qwen_forward, self.model)
        except Exception:
            pass

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
        if hasattr(module, "_qwen_te_shard") and module._qwen_te_shard == self._shard_order[-1]:
            hidden = output[0] if isinstance(output, tuple) else output
            self._last_hidden_state = hidden.detach()
            
        evict_module(module)
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

        # Use the inner language model directly for pure-text prompts.
        # The top-level Qwen2.5-VL wrapper routes through multimodal masking logic,
        # which can inherit meta-device state from the vision stack even when no
        # image inputs are involved.
        out = self.model.model.language_model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
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
        cache_to_ram: bool = False,
        prefetch: bool = True,
        max_length: int = 512,
        is_edit_model: bool = False,
        **kwargs
    ) -> "Qwen2_5_VLForConditionalGenerationStreamer":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_dir = Path(model_dir)

        print("  [TE 1/3] Initializing LiveSeeker on Qwen text encoder weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        print(f"    Found {len(seeker.weight_map)} tensors.")

        print("  [TE 2/3] Instantiating Qwen2_5_VLForConditionalGeneration on meta device ...")
        with default_dtype(dtype), init_empty_weights():
            from transformers import Qwen2_5_VLConfig
            cfg = Qwen2_5_VLConfig.from_pretrained(str(model_dir))
            model = Qwen2_5_VLForConditionalGeneration(cfg)
        model.eval()

        # Move non-meta buffers (those with real storage) to the target device.
        # Do NOT attempt to move parameters that are still on 'meta' (they have no data).
        for buf_name, buf in model.named_buffers():
            if buf is None:
                continue
            dev = getattr(buf, 'device', None)
            if dev is None:
                continue
            # skip buffers that are still meta (no data yet)
            if dev.type == "meta":
                continue
            # move buffers that are on a different device to the target device
            if dev.type != device:
                if buf.is_floating_point():
                    set_module_tensor_to_device(model, buf_name, device, value=buf, dtype=dtype)
                else:
                    set_module_tensor_to_device(model, buf_name, device, value=buf)

        print("  [TE 3/3] Loading resident Qwen text encoder tensors ...")
        resident_keys = _get_resident_keys(seeker, is_edit_model=is_edit_model)
        resident_sd = seeker.get_tensors(resident_keys, device="cpu", dtype=dtype)
        
        mapped_sd = {map_qwen_key(k): v for k, v in resident_sd.items()}
        cpu_sd = {k: v for k, v in mapped_sd.items() if "embed_tokens" in k}
        gpu_sd = {k: v for k, v in mapped_sd.items() if k not in cpu_sd}
        
        if cpu_sd:
            place_tensors(model, cpu_sd, "cpu", dtype)
            from weellm.memory import pin_module_to_cpu
            pin_module_to_cpu(model, "model.language_model.embed_tokens")
            
        if gpu_sd:
            place_tensors(model, gpu_sd, device, dtype)
            
        del resident_sd, mapped_sd, cpu_sd, gpu_sd

        # Some Qwen components (most notably lm_head) can remain on the target
        # device but keep their original float32 dtype after loading. Normalize
        # all resident floating-point parameters to the requested compute dtype
        # so mixed-dtype matmuls do not fail in the prompt encoder.
        for param_name, param in model.named_parameters():
            if param is None:
                continue
            dev = getattr(param, "device", None)
            if dev is None or dev.type == "meta":
                continue
            if param.is_floating_point() and param.dtype != dtype:
                set_module_tensor_to_device(model, param_name, device, value=param, dtype=dtype)

        # Qwen's tied lm_head can still retain float32 after the generic pass above.
        # Cast it explicitly so the prompt encoder and language model agree on dtype.
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
            try:
                model.lm_head.to(device=device, dtype=dtype)
            except Exception:
                pass
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "lm_head"):
            try:
                model.model.language_model.lm_head.to(device=device, dtype=dtype)
            except Exception:
                pass

        # Re-tie the head after loading, so the shared output projection follows
        # the already-loaded embed_tokens tensor instead of preserving its init dtype.
        try:
            model.tie_weights()
        except Exception:
            pass

        # Override the runtime dtype reported by the Qwen model instance so
        # Diffusers prompt encoding uses bfloat16 instead of a meta float32 param.
        try:
            orig_cls = model.__class__
            class _WeeLLMQwenDTypeModel(orig_cls):
                @property
                def dtype(self):
                    return getattr(self, "_weellm_forced_dtype", dtype)

            model.__class__ = _WeeLLMQwenDTypeModel
            model._weellm_forced_dtype = dtype
            if hasattr(model, "model") and hasattr(model.model, "language_model"):
                lm_cls = model.model.language_model.__class__
                class _WeeLLMQwenLanguageModelDType(lm_cls):
                    @property
                    def dtype(self):
                        return getattr(self, "_weellm_forced_dtype", dtype)
                model.model.language_model.__class__ = _WeeLLMQwenLanguageModelDType
                model.model.language_model._weellm_forced_dtype = dtype
        except Exception:
            pass

        clean_memory(device)

        # Materialize any small control buffers that remain on 'meta' (e.g., cache position,
        # cumulative length) by creating a zero tensor with the same shape/dtype on the target device.
        # This avoids downstream code trying to `.to(device=meta_device)` which fails.
        for buf_name, buf in model.named_buffers():
            if buf is None:
                continue
            dev = getattr(buf, 'device', None)
            if dev is None or dev.type != 'meta':
                continue
            # create a device-backed tensor with the same shape/dtype
            try:
                shape = tuple(buf.shape)
            except Exception:
                continue
            try:
                new_buf = torch.zeros(shape, dtype=buf.dtype, device=device)
                set_module_tensor_to_device(model, buf_name, device, value=new_buf)
            except Exception:
                # If we can't safely materialize, skip — the buffer may be large or complex.
                pass

        # Additionally, some models store control tensors as plain attributes (not registered buffers).
        # Walk modules and replace any plain `torch.Tensor` attributes that are on 'meta' with
        # device-backed zero tensors. Skip `nn.Parameter` to avoid touching model weights.
        for mod in model.modules():
            for attr_name, attr_val in list(mod.__dict__.items()):
                try:
                    if isinstance(attr_val, torch.Tensor) and not isinstance(attr_val, nn.Parameter):
                        dev = getattr(attr_val, 'device', None)
                        if dev is not None and dev.type == 'meta':
                            shape = tuple(attr_val.shape) if hasattr(attr_val, 'shape') else None
                            if shape is None:
                                continue
                            try:
                                new_t = torch.zeros(shape, dtype=attr_val.dtype, device=device)
                                setattr(mod, attr_name, new_t)
                            except Exception:
                                # best-effort; skip attributes we can't materialize
                                continue
                except Exception:
                    continue

        # Deep-scan object attributes (best-effort) to find meta tensors inside helper objects
        # such as Cache/past_key_values which are not nn.Modules and weren't covered above.
        def _scan_and_materialize(root_obj, max_depth=3):
            seen = set()
            queue = [(root_obj, 0, None, None)]  # (obj, depth, parent_obj, attr_name_or_index)

            while queue:
                obj, depth, parent, name = queue.pop(0)
                if id(obj) in seen or depth > max_depth:
                    continue
                seen.add(id(obj))

                # Inspect tensors directly
                if isinstance(obj, torch.Tensor):
                    try:
                        dev = getattr(obj, 'device', None)
                        if dev is not None and dev.type == 'meta':
                            # attempt to replace on parent
                            if parent is not None and name is not None:
                                try:
                                    shape = tuple(obj.shape)
                                    new_t = torch.zeros(shape, dtype=obj.dtype, device=device)
                                    if isinstance(parent, dict):
                                        parent[name] = new_t
                                    elif isinstance(name, int) and isinstance(parent, (list, tuple)):
                                        if isinstance(parent, list):
                                            parent[name] = new_t
                                    else:
                                        setattr(parent, name, new_t)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    continue

                # Iterate attributes for non-primitive objects
                try:
                    if isinstance(obj, dict):
                        for k, v in list(obj.items()):
                            queue.append((v, depth + 1, obj, k))
                    elif isinstance(obj, (list, tuple, set)):
                        for idx, v in enumerate(list(obj)):
                            queue.append((v, depth + 1, obj, idx))
                    else:
                        for attr, val in list(getattr(obj, '__dict__', {}).items()):
                            queue.append((val, depth + 1, obj, attr))
                except Exception:
                    continue

        try:
            _scan_and_materialize(model, max_depth=4)
        except Exception:
            pass

        layer_count = len(model.model.language_model.layers)
        print(f"    -> {layer_count} Qwen layers will stream on-demand.")
        report_memory("After Qwen text encoder init")

        instance = cls(
            model=model,
            seeker=seeker,
            layer_count=layer_count,
            tokenizer=tokenizer,
            device=device,
            cache_to_ram=cache_to_ram,
            dtype=dtype,
            prefetch=prefetch,
            max_length=max_length,
        )
        return instance
