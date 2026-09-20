import os
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import Gemma2Model, Gemma2Config, Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
import threading
from concurrent.futures import ThreadPoolExecutor
from weellm.io.memory import evict_module, place_tensors
from weellm.io.seeker import get_seeker
import logging

logger = logging.getLogger("weellm")

class Gemma4UnifiedForConditionalGenerationStreamer:
    """
    Streams Gemma2 9B/12B layer-by-layer for LTX-2.5 Text Encoder.
    """
    def __init__(self, model, seeker, device="cuda", dtype=torch.bfloat16, prefetch=True):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch
        
        uses_nested = any(k.startswith("model.language_model.layers.") for k in self.seeker.weight_map)
        self.shard_prefix = "model.language_model.layers" if uses_nested else "model.layers"
        self.layer_count = len(model.layers)
        self._shard_order = [f"{self.shard_prefix}.{i}" for i in range(self.layer_count)]
        
        self._prefetch_depth = 0
        if self.prefetch:
            try:
                import psutil
                _RAM_SAFETY_BYTES = 2 * 1024 * 1024 * 1024
                _MAX_PREFETCH_DEPTH = 6
                sample_keys = [k for k in self.seeker.weight_map if k.startswith(self._shard_order[0] + ".")]
                layer_bytes = self.seeker.get_block_bytes(sample_keys)
                available = psutil.virtual_memory().available
                usable = max(0, available - _RAM_SAFETY_BYTES)
                
                if usable <= 0:
                    logger.warning("[TE Streamer] Available RAM (%.1f GB) below safety threshold. Disabling TE prefetch.", available / 1e9)
                    self._prefetch_depth = 0
                else:
                    self._prefetch_depth = max(1, min(_MAX_PREFETCH_DEPTH, int(usable // layer_bytes)))
                    logger.info(
                        "[TE Streamer] Adaptive prefetch depth: %d (layer=%.0f MB, usable RAM=%.1f GB)",
                        self._prefetch_depth, layer_bytes / 1e6, usable / 1e9,
                    )
            except Exception as e:
                self._prefetch_depth = 1
                logger.debug("[TE Streamer] Failed to calculate adaptive prefetch depth, defaulting to 1.")

        if self._prefetch_depth > 0:
            self._executor = ThreadPoolExecutor(max_workers=self._prefetch_depth)
        else:
            self._executor = None
            
        self._prefetch_futures = {}
        self._lock = threading.Lock()
        
        if self._executor is not None:
            for i in range(min(self._prefetch_depth, self.layer_count)):
                shard_name = self._shard_order[i]
                layer_keys = [k for k in self.seeker.weight_map if k.startswith(shard_name + ".")]
                self._prefetch_futures[shard_name] = self._executor.submit(
                    self.seeker.get_tensors, layer_keys, "cpu", self.dtype
                )
        
        self._install_hooks()

    def _get_resident_keys(self):
        resident = []
        for k in self.seeker.weight_map.keys():
            if not (k.startswith("model.layers.") or k.startswith("model.language_model.layers.")):
                resident.append(k)
        return resident

    def _install_hooks(self):
        for i, shard_name in enumerate(self._shard_order):
            layer = self.model.layers[i]
            layer._gemma_te_shard = shard_name
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, args):
        import time
        t0 = time.time()
        shard_name = module._gemma_te_shard
        pos = int(shard_name.split(".")[-1])
        layer_keys = [k for k in self.seeker.weight_map if k.startswith(shard_name + ".")]
        
        fut = None
        with self._lock:
            fut = self._prefetch_futures.pop(shard_name, None)
            
        if fut is not None:
            sd = fut.result()
        else:
            sd = self.seeker.get_tensors(layer_keys, device="cpu", dtype=self.dtype)
                
        t1 = time.time()
        
        sd = {k: v.to(self.device, non_blocking=True) for k, v in sd.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                mapped_sd[k[len("model.language_model."):]] = v
            elif k.startswith("model."):
                mapped_sd[k[len("model."):]] = v
            else:
                mapped_sd[k] = v
                
        for mapped_k, mapped_v in mapped_sd.items():
            try:
                place_tensors(self.model, {mapped_k: mapped_v}, self.device, self.dtype, skip_errors=False)
            except Exception as e:
                if "layer_scalar" not in mapped_k:
                    logger.error(f"FAILED TO PLACE {mapped_k}: {repr(e)}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del sd
                    
        t2 = time.time()
        logger.info(
            "    [TE Streamer] %s (%d/%d): Disk/Wait=%.3fs | H2D+Apply=%.3fs",
            shard_name, pos + 1, self.layer_count, t1 - t0, t2 - t1,
        )
        module._weellm_t_compute_start = time.time()
        
        if self._executor is not None:
            next_pos = pos + self._prefetch_depth
            if next_pos < self.layer_count:
                next_name = self._shard_order[next_pos]
                next_keys = [k for k in self.seeker.weight_map if k.startswith(next_name + ".")]
                with self._lock:
                    if next_name not in self._prefetch_futures:
                        self._prefetch_futures[next_name] = self._executor.submit(
                            self.seeker.get_tensors, next_keys, "cpu", self.dtype
                        )

    def _post_hook(self, module, args, output):
        import time
        t_end = time.time()
        t_start = getattr(module, "_weellm_t_compute_start", t_end)
        shard_name = module._gemma_te_shard
        logger.info("    [TE Streamer] %s: GPU Compute=%.3fs (Offloading...)", shard_name, t_end - t_start)
        
        evict_module(module)
        return output

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
        import json
        import os
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
            
        if "text_config" in cfg_dict and "use_bidirectional_attention" in cfg_dict["text_config"]:
            cfg_dict["text_config"]["use_bidirectional_attention"] = False
            
        config = Gemma3Config.from_dict(cfg_dict)
        text_config = config.text_config
        
        # Use the config vocab size exactly as provided in the config.json
        # text_config.vocab_size = 262144
        
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
                    
        model.eval()

        # Fix PyTorch meta buffers that lose their values during init_empty_weights()
        logger.info("  Re-initializing rotary embeddings and scale buffers on GPU ...")
        # 1. Re-initialize RoPE so all inv_freq buffers are calculated with real data
        model.rotary_emb = type(model.rotary_emb)(text_config).to(device)
        
        # Patch rotary embedding for full_attention to use dim=512 ON DEVICE!
        # Must correctly implement 'proportional' RoPE for dim 512 to avoid NaNs on long sequences
        base = text_config.rope_parameters["full_attention"]["rope_theta"]
        head_dim = 512
        rope_proportion = text_config.rope_parameters["full_attention"]["partial_rotary_factor"]
        rope_angles = int(rope_proportion * head_dim // 2)
        
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
        model.rotary_emb.register_buffer("full_attention_inv_freq", inv_freq, persistent=False)
        model.rotary_emb.register_buffer("full_attention_original_inv_freq", inv_freq.clone(), persistent=False)
        
        # 2. Fix embed_scale which also lost its value on the meta device
        if hasattr(model.embed_tokens, "embed_scale"):
            model.embed_tokens.embed_scale = torch.tensor(
                text_config.hidden_size**0.5, dtype=torch.float32, device=device
            )
            
        # 3. For any other remaining meta buffers (e.g. padding/mask buffers), just zero them out safely
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
        
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                mapped_sd[k[len("model.language_model."):]] = v
            elif k.startswith("model."):
                mapped_sd[k[len("model."):]] = v
            else:
                mapped_sd[k] = v
                
        for mapped_k, mapped_v in mapped_sd.items():
            try:
                place_tensors(model, {mapped_k: mapped_v}, device, dtype, skip_errors=False)
            except Exception as e:
                logger.error(f"[TE Streamer] Failed to place resident tensor {mapped_k}: {repr(e)}")
        del sd
        
        return streamer
