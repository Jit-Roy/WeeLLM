"""
gguf_seek.py -- GGUF tensor reader for WeeLLM layer-streaming.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger("weellm")

# FLUX double/single block remapping (llama.cpp convention)
_FLUX_KEY_MAP = {
    # Double blocks
    "blk.{i}.img_attn_q.weight":           "transformer_blocks.{i}.attn.to_q.weight",
    "blk.{i}.img_attn_k.weight":           "transformer_blocks.{i}.attn.to_k.weight",
    "blk.{i}.img_attn_v.weight":           "transformer_blocks.{i}.attn.to_v.weight",
    "blk.{i}.img_attn_proj.weight":        "transformer_blocks.{i}.attn.to_out.0.weight",
    "blk.{i}.img_mlp.0.weight":            "transformer_blocks.{i}.ff.net.0.proj.weight",
    "blk.{i}.img_mlp.2.weight":            "transformer_blocks.{i}.ff.net.2.weight",
    "blk.{i}.img_mod.lin.weight":          "transformer_blocks.{i}.norm1.linear.weight",
    "blk.{i}.img_mod.lin.bias":            "transformer_blocks.{i}.norm1.linear.bias",
    "blk.{i}.img_norm1.weight":            "transformer_blocks.{i}.norm1.norm.weight",
    "blk.{i}.img_norm2.weight":            "transformer_blocks.{i}.norm1_context.norm.weight",
    "blk.{i}.txt_attn_q.weight":           "transformer_blocks.{i}.attn.add_q_proj.weight",
    "blk.{i}.txt_attn_k.weight":           "transformer_blocks.{i}.attn.add_k_proj.weight",
    "blk.{i}.txt_attn_v.weight":           "transformer_blocks.{i}.attn.add_v_proj.weight",
    "blk.{i}.txt_attn_proj.weight":        "transformer_blocks.{i}.attn.to_add_out.weight",
    "blk.{i}.txt_mlp.0.weight":            "transformer_blocks.{i}.ff_context.net.0.proj.weight",
    "blk.{i}.txt_mlp.2.weight":            "transformer_blocks.{i}.ff_context.net.2.weight",
    "blk.{i}.txt_mod.lin.weight":          "transformer_blocks.{i}.norm1_context.linear.weight",
    "blk.{i}.txt_mod.lin.bias":            "transformer_blocks.{i}.norm1_context.linear.bias",
    "blk.{i}.txt_norm1.weight":            "transformer_blocks.{i}.norm1.norm.weight",
    "blk.{i}.txt_norm2.weight":            "transformer_blocks.{i}.norm2_context.weight",
    # Single blocks
    "single_blk.{i}.linear1.weight":       "single_transformer_blocks.{i}.attn.to_q.weight",
    "single_blk.{i}.linear2.weight":       "single_transformer_blocks.{i}.proj_out.weight",
    "single_blk.{i}.pre_norm.weight":      "single_transformer_blocks.{i}.norm.linear.weight",
    "single_blk.{i}.pre_norm.bias":        "single_transformer_blocks.{i}.norm.linear.bias",
}

# LLAMA / Qwen / Mistral / Gemma remapping
_LLAMA_KEY_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_q.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_k.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_v.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.attn_output.bias": "model.layers.{i}.self_attn.o_proj.bias",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
}

# Tensors in BFL/ComfyUI GGUF format that need their two output-half rows
# swapped before use with Diffusers.  During the BFL→Diffusers conversion the
# script applies swap_scale_shift() to these weights so the diffusers model
# expects [shift | scale] order, while the original GGUF has [scale | shift].
_SWAP_SCALE_SHIFT_GGUF_KEYS: frozenset = frozenset({
    "final_layer.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias",
})

# FLUX ComfyUI -> Diffusers remapping (ComfyUI convention)
_COMFY_FLUX_KEY_MAP = {
    # Double blocks
    "double_blocks.{i}.img_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.to_q.weight", 0, 3),
        ("transformer_blocks.{i}.attn.to_k.weight", 1, 3),
        ("transformer_blocks.{i}.attn.to_v.weight", 2, 3),
    ],
    "double_blocks.{i}.txt_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.add_q_proj.weight", 0, 3),
        ("transformer_blocks.{i}.attn.add_k_proj.weight", 1, 3),
        ("transformer_blocks.{i}.attn.add_v_proj.weight", 2, 3),
    ],
    "double_blocks.{i}.img_attn.proj.weight": "transformer_blocks.{i}.attn.to_out.0.weight",
    "double_blocks.{i}.txt_attn.proj.weight": "transformer_blocks.{i}.attn.to_add_out.weight",
    "double_blocks.{i}.img_mlp.0.weight": "transformer_blocks.{i}.ff.linear_in.weight",
    "double_blocks.{i}.img_mlp.0.bias": "transformer_blocks.{i}.ff.linear_in.bias",
    "double_blocks.{i}.img_mlp.2.weight": "transformer_blocks.{i}.ff.linear_out.weight",
    "double_blocks.{i}.img_mlp.2.bias": "transformer_blocks.{i}.ff.linear_out.bias",
    "double_blocks.{i}.txt_mlp.0.weight": "transformer_blocks.{i}.ff_context.linear_in.weight",
    "double_blocks.{i}.txt_mlp.0.bias": "transformer_blocks.{i}.ff_context.linear_in.bias",
    "double_blocks.{i}.txt_mlp.2.weight": "transformer_blocks.{i}.ff_context.linear_out.weight",
    "double_blocks.{i}.txt_mlp.2.bias": "transformer_blocks.{i}.ff_context.linear_out.bias",
    
    "double_blocks.{i}.img_mod.lin.weight": "transformer_blocks.{i}.norm1.linear.weight",
    "double_blocks.{i}.img_mod.lin.bias": "transformer_blocks.{i}.norm1.linear.bias",
    "double_blocks.{i}.txt_mod.lin.weight": "transformer_blocks.{i}.norm1_context.linear.weight",
    "double_blocks.{i}.txt_mod.lin.bias": "transformer_blocks.{i}.norm1_context.linear.bias",

    "double_blocks.{i}.img_attn.norm.key_norm.scale": "transformer_blocks.{i}.attn.norm_k.weight",
    "double_blocks.{i}.img_attn.norm.query_norm.scale": "transformer_blocks.{i}.attn.norm_q.weight",
    "double_blocks.{i}.txt_attn.norm.key_norm.scale": "transformer_blocks.{i}.attn.norm_added_k.weight",
    "double_blocks.{i}.txt_attn.norm.query_norm.scale": "transformer_blocks.{i}.attn.norm_added_q.weight",

    # Single blocks
    "single_blocks.{i}.linear1.weight": "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight",
    "single_blocks.{i}.linear2.weight": "single_transformer_blocks.{i}.attn.to_out.weight",
    "single_blocks.{i}.modulation.lin.weight": "single_transformer_blocks.{i}.norm.linear.weight",
    "single_blocks.{i}.modulation.lin.bias": "single_transformer_blocks.{i}.norm.linear.bias",
    "single_blocks.{i}.norm.key_norm.scale": "single_transformer_blocks.{i}.attn.norm_k.weight",
    "single_blocks.{i}.norm.query_norm.scale": "single_transformer_blocks.{i}.attn.norm_q.weight",
    
    # Top-level Modulations & Time
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.in_layer.bias": "time_guidance_embed.timestep_embedder.linear_1.bias",
    "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "time_in.out_layer.bias": "time_guidance_embed.timestep_embedder.linear_2.bias",
    "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
    "guidance_in.in_layer.bias": "time_guidance_embed.guidance_embedder.linear_1.bias",
    "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
    "guidance_in.out_layer.bias": "time_guidance_embed.guidance_embedder.linear_2.bias",
    "vector_in.in_layer.weight": "context_embedder.weight",
    "vector_in.in_layer.bias": "context_embedder.bias",
    "txt_in.weight": "context_embedder.weight",
    "txt_in.bias": "context_embedder.bias",
    "img_in.weight": "x_embedder.weight",
    "img_in.bias": "x_embedder.bias",
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
}

def _build_remap_fn(gguf_keys: List[str], arch: str = "unknown"):
    # Check if llama.cpp format
    if any(k.startswith("blk.") or k.startswith("single_blk.") for k in gguf_keys):
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("blk", "single_blk") and len(parts) > 1:
                try: max_i = max(max_i, int(parts[1]))
                except ValueError: pass
        
        remap = {}
        tmpl_map = _LLAMA_KEY_MAP if arch in ("llama", "qwen2", "qwen2.5", "qwen3", "gemma2", "mistral", "chatglm", "glm") else _FLUX_KEY_MAP
        for tmpl_src, tmpl_dst in tmpl_map.items():
            for i in range(max_i + 1):
                remap[tmpl_src.replace("{i}", str(i))] = [(tmpl_dst.replace("{i}", str(i)), None)]
                
        def _remap(name: str):
            return remap.get(name, [(name, None)])
        logger.info("[GGUFSeeker] llama.cpp key naming detected — remapping to Diffusers convention.")
        return _remap
        
    # Check if ComfyUI format
    elif any(k.startswith("double_blocks.") for k in gguf_keys):
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("double_blocks", "single_blocks") and len(parts) > 1:
                try: max_i = max(max_i, int(parts[1]))
                except ValueError: pass
                
        remap = {}
        for tmpl_src, tmpl_dst in _COMFY_FLUX_KEY_MAP.items():
            for i in range(max_i + 1):
                src = tmpl_src.replace("{i}", str(i))
                if isinstance(tmpl_dst, list):
                    remap[src] = [(dst.replace("{i}", str(i)), (split_idx, total_splits)) for dst, split_idx, total_splits in tmpl_dst]
                else:
                    remap[src] = [(tmpl_dst.replace("{i}", str(i)), None)]
                    
        def _remap(name: str):
            return remap.get(name, [(name, None)])
        logger.info("[GGUFSeeker] ComfyUI key naming detected — remapping to Diffusers convention.")
        return _remap

    else:
        def _remap(name: str):
            return [(name, None)]
        return _remap

class GGUFSeeker:
    def __init__(self, gguf_path: Path) -> None:
        try:
            import gguf as _gguf_lib
        except ImportError:
            raise ImportError("gguf>=0.13.0 is required.")

        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        logger.info("[GGUFSeeker] Opening %s ...", self.gguf_path.name)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self._reader = _gguf_lib.GGUFReader(str(self.gguf_path))

        self._tensor_meta: Dict[str, Tuple] = {}
        raw_names = [t.name for t in self._reader.tensors]
        
        arch = self._get_arch()
        remap = _build_remap_fn(raw_names, arch)

        import gguf as _gguf_lib
        for tensor in self._reader.tensors:
            orig_shape = self._get_orig_shape(tensor.name)
            if orig_shape is None:
                orig_shape = tuple(int(v) for v in reversed(tensor.shape))

            diffusers_entries = remap(tensor.name)
            for diffusers_name, slice_info in diffusers_entries:
                self._tensor_meta[diffusers_name] = (
                    tensor.tensor_type,
                    orig_shape,
                    tensor.data,
                    slice_info,
                    tensor.name  # keep track of original name for caching
                )

        self.weight_map: Dict[str, str] = {k: self.gguf_path.name for k in self._tensor_meta}
        logger.info("[GGUFSeeker] Loaded %d tensors (arch=%s)", len(self._tensor_meta), arch)

    def _get_arch(self) -> str:
        try:
            field = self._reader.get_field("general.architecture")
            if field is not None and field.types:
                import gguf as _gguf_lib
                if field.types[0] == _gguf_lib.GGUFValueType.STRING:
                    return str(field.parts[field.data[-1]], encoding="utf-8")
        except Exception:
            pass
        return "unknown"

    def _get_orig_shape(self, tensor_name: str):
        try:
            import gguf as _gguf_lib
            field_key = f"comfy.gguf.orig_shape.{tensor_name}"
            field     = self._reader.get_field(field_key)
            if field is None: return None
            if (len(field.types) == 2 and field.types[0] == _gguf_lib.GGUFValueType.ARRAY and field.types[1] == _gguf_lib.GGUFValueType.INT32):
                return tuple(int(field.parts[part_idx][0]) for part_idx in field.data)
        except Exception:
            pass
        return None

    def get_tensors(self, keys: List[str], device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        import gguf as _gguf_lib
        from weellm.gguf_dequant import dequantize_tensor, TORCH_COMPATIBLE_QTYPES

        result: Dict[str, torch.Tensor] = {}
        target_dtype = dtype if dtype is not None else torch.bfloat16
        
        # Cache dequantized tensors if multiple slices come from the same original tensor
        _dequantized_cache = {}

        for key in keys:
            if key not in self._tensor_meta:
                raise KeyError(f"Tensor '{key}' not found in GGUF file.")

            qtype, shape, raw_data_np, slice_info, orig_name = self._tensor_meta[key]

            if orig_name in _dequantized_cache:
                t = _dequantized_cache[orig_name]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                    raw_torch = torch.from_numpy(raw_data_np)

                if qtype in TORCH_COMPATIBLE_QTYPES:
                    if qtype == _gguf_lib.GGMLQuantizationType.F32:
                        t = raw_torch.view(torch.float32).reshape(shape)
                    elif qtype == _gguf_lib.GGMLQuantizationType.F16:
                        t = raw_torch.view(torch.float16).reshape(shape)
                    else:
                        t = raw_torch.reshape(shape)
                    if t.is_floating_point() and t.dtype != target_dtype:
                        t = t.to(target_dtype)
                else:
                    # Move raw bytes to a GPU device temporarily to massively accelerate dequantization
                    temp_device = "cuda" if torch.cuda.is_available() else device
                    t = dequantize_tensor(raw_torch.to(temp_device, non_blocking=True), qtype, shape, dtype=target_dtype)
                    if device == "cpu" and temp_device != "cpu":
                        t = t.to("cpu") # MUST BE SYNCHRONOUS to prevent race condition with H2D stream
                    
                _dequantized_cache[orig_name] = t
                
            if slice_info is not None:
                split_idx, total_splits = slice_info
                # Split along dim 0
                chunk_size = t.shape[0] // total_splits
                t = t[split_idx * chunk_size : (split_idx + 1) * chunk_size, ...]

            # Need to clone if sliced, otherwise it keeps the whole tensor in memory!
            if slice_info is not None:
                t = t.clone()

            # Apply swap_scale_shift for tensors whose output halves are stored
            # in [scale | shift] order in the original BFL/GGUF format but must
            # be [shift | scale] for the diffusers model (norm_out.linear, etc.).
            if orig_name in _SWAP_SCALE_SHIFT_GGUF_KEYS:
                half = t.shape[0] // 2
                t = torch.cat([t[half:], t[:half]], dim=0).contiguous()

            result[key] = t.to(device=device)

        return result

    def get_block_bytes(self, keys: List[str]) -> int:
        import math
        total = 0
        for key in keys:
            if key not in self._tensor_meta: continue
            _, shape, _, slice_info, _ = self._tensor_meta[key]
            n_elements = math.prod(shape) if shape else 1
            if slice_info is not None:
                n_elements = n_elements // slice_info[1]
            total += n_elements * 2
        return total
