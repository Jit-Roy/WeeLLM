import os
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import Gemma2Model, Gemma2Config, Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
from weellm.models.base_streamer import BaseTransformerStreamer
from weellm.seeker import get_seeker
import logging

logger = logging.getLogger("weellm")

class LTX2GemmaStreamer(BaseTransformerStreamer):
    """
    Streams Gemma2 9B/12B layer-by-layer for LTX-2.5 Text Encoder.
    """
    def _get_shard_order(self):
        order = []
        # Detect which key prefix the checkpoint actually uses
        uses_nested = any(
            k.startswith("model.language_model.layers.") for k in self.seeker.weight_map.keys()
        )
        shard_prefix = "model.language_model.layers" if uses_nested else "model.layers"

        layers = self.model.layers  # Gemma3TextModel exposes .layers directly

        for i, layer in enumerate(layers):
            shard_name = f"{shard_prefix}.{i}"
            order.append((shard_name, layer))
        return order

    def _get_resident_keys(self):
        resident = []
        for k in self.seeker.weight_map.keys():
            # Exclude per-layer shard keys; keep all resident (embed, norms, etc.)
            if not (k.startswith("model.layers.") or k.startswith("model.language_model.layers.")):
                resident.append(k)
        return resident

    def _get_layer_keys(self, shard_name: str):
        return [k for k in self.seeker.weight_map if k.startswith(shard_name + ".")]

    def apply_state_dict(self, state_dict, skip_errors=False):
        from weellm.memory import place_tensors
        mapped_sd = {}
        for k, v in state_dict.items():
            # Checkpoint uses model.language_model.* or model.* prefixes.
            # Gemma3TextModel is the bare text model with NO .language_model sub-module,
            # so we always strip down to the attribute name the model actually has.
            if k.startswith("model.language_model."):
                # e.g. model.language_model.embed_tokens.weight -> embed_tokens.weight
                mapped_sd[k[len("model.language_model."):]] = v
            elif k.startswith("model."):
                # e.g. model.norm.weight -> norm.weight
                mapped_sd[k[len("model."):]] = v
            else:
                mapped_sd[k] = v

        for mapped_k, mapped_v in mapped_sd.items():
            try:
                place_tensors(self.model, {mapped_k: mapped_v}, self.device, self.dtype, skip_errors=False)
            except Exception as e:
                if "layer_scalar" not in mapped_k:
                    logger.error(f"FAILED TO PLACE {mapped_k}: {repr(e)}")

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        device="cuda",
        dtype=torch.bfloat16,
        prefetch=True,
        cache_to_ram=False,
    ):
        logger.info("Step 1/3 -- Initializing LiveSeeker on Gemma4-12B weights ...")
        seeker = get_seeker(str(model_dir), cache_to_ram=cache_to_ram)
        
        logger.info("  Instantiating Gemma3TextModel on meta device ...")
        # Exact LTX-2.5 Text Encoder (Gemma 3 12B architecture):
        config = Gemma3Config.from_pretrained("google/gemma-3-12b-it")
        text_config = config.text_config
        
        # Crucial fix: The original Gemma 3 config has vocab_size 262208, 
        # but the LTX-2.5 safetensors checkpoint has exactly 262144 embeddings.
        text_config.vocab_size = 262144
        
        with init_empty_weights():
            model = Gemma3TextModel(text_config)
            
            # Monkey-patch Gemma 4 (LTX-2.5) specific layer dimensions:
            # Gemma 3 Text Model defines uniform head counts for all layers.
            # However, LTX-2.5 uses 32 Q-heads and 2 KV-heads for full_attention layers,
            # and 16 Q-heads and 8 KV-heads for sliding_attention layers.
            hidden_size = text_config.hidden_size
            head_dim = getattr(text_config, "head_dim", hidden_size // text_config.num_attention_heads)
            for i, layer in enumerate(model.layers):
                if text_config.layer_types[i] == "full_attention":
                    # Patch for 16 q_heads, 1 kv_head, head_dim = 512
                    layer.self_attn.head_dim = 512
                    layer.self_attn.num_key_value_groups = 16 // 1
                    layer.self_attn.q_proj = nn.Linear(hidden_size, 16 * 512, bias=text_config.attention_bias)
                    layer.self_attn.k_proj = nn.Linear(hidden_size, 1 * 512, bias=text_config.attention_bias)
                    layer.self_attn.v_proj = nn.Linear(hidden_size, 1 * 512, bias=text_config.attention_bias)
                    layer.self_attn.o_proj = nn.Linear(16 * 512, hidden_size, bias=text_config.attention_bias)
                    layer.self_attn.q_norm = type(layer.self_attn.q_norm)(dim=512, eps=text_config.rms_norm_eps)
                    layer.self_attn.k_norm = type(layer.self_attn.k_norm)(dim=512, eps=text_config.rms_norm_eps)
                    
            # Patch rotary embedding for full_attention to use dim=512
            base = text_config.rope_parameters["full_attention"]["rope_theta"]
            inv_freq = 1.0 / (base ** (torch.arange(0, 512, 2, dtype=torch.float32, device="meta") / 512.0))
            model.rotary_emb.register_buffer("full_attention_inv_freq", inv_freq, persistent=False)
            model.rotary_emb.register_buffer("full_attention_original_inv_freq", inv_freq.clone(), persistent=False)
                    
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type == "meta":
                try:
                    set_module_tensor_to_device(model, buf_name, device, value=torch.zeros_like(buf, device=device))
                except Exception:
                    pass

        logger.info("Step 2/3 -- Hooking streaming layers ...")
        streamer = cls(
            model=model,
            seeker=seeker,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )

        logger.info("Step 3/3 -- Loading resident tensors ...")
        resident_keys = streamer._get_resident_keys()
        sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(sd, skip_errors=True)
        del sd
        
        return streamer
