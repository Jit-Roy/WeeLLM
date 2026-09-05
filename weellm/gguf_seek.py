"""
gguf_seek.py -- GGUF tensor reader for WeeLLM layer-streaming.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger("weellm")

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

# SDXL ComfyUI -> Diffusers remapping (Condensed Template)
_SDXL_BLOCKS = [
    # (LDM Prefix, Diffusers Prefix, Num Transformers, Is Resnet)
    
    # Input Blocks
    ("input_blocks.1.0", "down_blocks.0.resnets.0", 0, True),
    ("input_blocks.2.0", "down_blocks.0.resnets.1", 0, True),
    ("input_blocks.4.0", "down_blocks.1.resnets.0", 0, True),
    ("input_blocks.4.1", "down_blocks.1.attentions.0", 2, False),
    ("input_blocks.5.0", "down_blocks.1.resnets.1", 0, True),
    ("input_blocks.5.1", "down_blocks.1.attentions.1", 2, False),
    ("input_blocks.7.0", "down_blocks.2.resnets.0", 0, True),
    ("input_blocks.7.1", "down_blocks.2.attentions.0", 10, False),
    ("input_blocks.8.0", "down_blocks.2.resnets.1", 0, True),
    ("input_blocks.8.1", "down_blocks.2.attentions.1", 10, False),
    
    # Middle Block
    ("middle_block.0", "mid_block.resnets.0", 0, True),
    ("middle_block.1", "mid_block.attentions.0", 10, False),
    ("middle_block.2", "mid_block.resnets.1", 0, True),
    
    # Output Blocks
    ("output_blocks.0.0", "up_blocks.0.resnets.0", 0, True),
    ("output_blocks.0.1", "up_blocks.0.attentions.0", 10, False),
    ("output_blocks.1.0", "up_blocks.0.resnets.1", 0, True),
    ("output_blocks.1.1", "up_blocks.0.attentions.1", 10, False),
    ("output_blocks.2.0", "up_blocks.0.resnets.2", 0, True),
    ("output_blocks.2.1", "up_blocks.0.attentions.2", 10, False),
    ("output_blocks.3.0", "up_blocks.1.resnets.0", 0, True),
    ("output_blocks.3.1", "up_blocks.1.attentions.0", 2, False),
    ("output_blocks.4.0", "up_blocks.1.resnets.1", 0, True),
    ("output_blocks.4.1", "up_blocks.1.attentions.1", 2, False),
    ("output_blocks.5.0", "up_blocks.1.resnets.2", 0, True),
    ("output_blocks.5.1", "up_blocks.1.attentions.2", 2, False),
    ("output_blocks.6.0", "up_blocks.2.resnets.0", 0, True),
    ("output_blocks.7.0", "up_blocks.2.resnets.1", 0, True),
    ("output_blocks.8.0", "up_blocks.2.resnets.2", 0, True),
]

_SDXL_RESNET_TEMPLATE = {
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
}

_SDXL_ATTN_TEMPLATE = {
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
    "proj_in.weight": "proj_in.weight",
    "proj_in.bias": "proj_in.bias",
    "proj_out.weight": "proj_out.weight",
    "proj_out.bias": "proj_out.bias",
    "transformer_blocks.{j}.norm1.weight": "transformer_blocks.{j}.norm1.weight",
    "transformer_blocks.{j}.norm1.bias": "transformer_blocks.{j}.norm1.bias",
    "transformer_blocks.{j}.attn1.to_q.weight": "transformer_blocks.{j}.attn1.to_q.weight",
    "transformer_blocks.{j}.attn1.to_k.weight": "transformer_blocks.{j}.attn1.to_k.weight",
    "transformer_blocks.{j}.attn1.to_v.weight": "transformer_blocks.{j}.attn1.to_v.weight",
    "transformer_blocks.{j}.attn1.to_out.0.weight": "transformer_blocks.{j}.attn1.to_out.0.weight",
    "transformer_blocks.{j}.attn1.to_out.0.bias": "transformer_blocks.{j}.attn1.to_out.0.bias",
    "transformer_blocks.{j}.norm2.weight": "transformer_blocks.{j}.norm2.weight",
    "transformer_blocks.{j}.norm2.bias": "transformer_blocks.{j}.norm2.bias",
    "transformer_blocks.{j}.attn2.to_q.weight": "transformer_blocks.{j}.attn2.to_q.weight",
    "transformer_blocks.{j}.attn2.to_k.weight": "transformer_blocks.{j}.attn2.to_k.weight",
    "transformer_blocks.{j}.attn2.to_v.weight": "transformer_blocks.{j}.attn2.to_v.weight",
    "transformer_blocks.{j}.attn2.to_out.0.weight": "transformer_blocks.{j}.attn2.to_out.0.weight",
    "transformer_blocks.{j}.attn2.to_out.0.bias": "transformer_blocks.{j}.attn2.to_out.0.bias",
    "transformer_blocks.{j}.norm3.weight": "transformer_blocks.{j}.norm3.weight",
    "transformer_blocks.{j}.norm3.bias": "transformer_blocks.{j}.norm3.bias",
    "transformer_blocks.{j}.ff.net.0.proj.weight": "transformer_blocks.{j}.ff.net.0.proj.weight",
    "transformer_blocks.{j}.ff.net.0.proj.bias": "transformer_blocks.{j}.ff.net.0.proj.bias",
    "transformer_blocks.{j}.ff.net.2.weight": "transformer_blocks.{j}.ff.net.2.weight",
    "transformer_blocks.{j}.ff.net.2.bias": "transformer_blocks.{j}.ff.net.2.bias",
}

_SD15_BLOCKS = [
    ('input_blocks.1.0', 'down_blocks.0.resnets.0', 0, True),
    ('input_blocks.1.1', 'down_blocks.0.attentions.0', 1, False),
    ('input_blocks.2.0', 'down_blocks.0.resnets.1', 0, True),
    ('input_blocks.2.1', 'down_blocks.0.attentions.1', 1, False),
    ('input_blocks.4.0', 'down_blocks.1.resnets.0', 0, True),
    ('input_blocks.4.1', 'down_blocks.1.attentions.0', 1, False),
    ('input_blocks.5.0', 'down_blocks.1.resnets.1', 0, True),
    ('input_blocks.5.1', 'down_blocks.1.attentions.1', 1, False),
    ('input_blocks.7.0', 'down_blocks.2.resnets.0', 0, True),
    ('input_blocks.7.1', 'down_blocks.2.attentions.0', 1, False),
    ('input_blocks.8.0', 'down_blocks.2.resnets.1', 0, True),
    ('input_blocks.8.1', 'down_blocks.2.attentions.1', 1, False),
    ('input_blocks.10.0', 'down_blocks.3.resnets.0', 0, True),
    ('input_blocks.11.0', 'down_blocks.3.resnets.1', 0, True),
    ('middle_block.0', 'mid_block.resnets.0', 0, True),
    ('middle_block.1', 'mid_block.attentions.0', 1, False),
    ('middle_block.2', 'mid_block.resnets.1', 0, True),
    ('output_blocks.0.0', 'up_blocks.0.resnets.0', 0, True),
    ('output_blocks.1.0', 'up_blocks.0.resnets.1', 0, True),
    ('output_blocks.2.0', 'up_blocks.0.resnets.2', 0, True),
    ('output_blocks.3.0', 'up_blocks.1.resnets.0', 0, True),
    ('output_blocks.3.1', 'up_blocks.1.attentions.0', 1, False),
    ('output_blocks.4.0', 'up_blocks.1.resnets.1', 0, True),
    ('output_blocks.4.1', 'up_blocks.1.attentions.1', 1, False),
    ('output_blocks.5.0', 'up_blocks.1.resnets.2', 0, True),
    ('output_blocks.5.1', 'up_blocks.1.attentions.2', 1, False),
    ('output_blocks.6.0', 'up_blocks.2.resnets.0', 0, True),
    ('output_blocks.6.1', 'up_blocks.2.attentions.0', 1, False),
    ('output_blocks.7.0', 'up_blocks.2.resnets.1', 0, True),
    ('output_blocks.7.1', 'up_blocks.2.attentions.1', 1, False),
    ('output_blocks.8.0', 'up_blocks.2.resnets.2', 0, True),
    ('output_blocks.8.1', 'up_blocks.2.attentions.2', 1, False),
    ('output_blocks.9.0', 'up_blocks.3.resnets.0', 0, True),
    ('output_blocks.9.1', 'up_blocks.3.attentions.0', 1, False),
    ('output_blocks.10.0', 'up_blocks.3.resnets.1', 0, True),
    ('output_blocks.10.1', 'up_blocks.3.attentions.1', 1, False),
    ('output_blocks.11.0', 'up_blocks.3.resnets.2', 0, True),
    ('output_blocks.11.1', 'up_blocks.3.attentions.2', 1, False),
]

_COMFY_SD15_BASE_MAP = {
    "model.diffusion_model.time_embed.0.weight": "time_embedding.linear_1.weight",
    "model.diffusion_model.time_embed.0.bias": "time_embedding.linear_1.bias",
    "model.diffusion_model.time_embed.2.weight": "time_embedding.linear_2.weight",
    "model.diffusion_model.time_embed.2.bias": "time_embedding.linear_2.bias",
    "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
    "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
    "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
    "model.diffusion_model.out.2.weight": "conv_out.weight",
    "model.diffusion_model.out.2.bias": "conv_out.bias",
    "model.diffusion_model.input_blocks.3.0.op.weight": "down_blocks.0.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias": "down_blocks.0.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight": "down_blocks.1.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias": "down_blocks.1.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.9.0.op.weight": "down_blocks.2.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.9.0.op.bias": "down_blocks.2.downsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.2.1.conv.weight": "up_blocks.0.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.2.1.conv.bias": "up_blocks.0.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "up_blocks.1.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias": "up_blocks.1.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "up_blocks.2.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias": "up_blocks.2.upsamplers.0.conv.bias",
}

_COMFY_SDXL_BASE_MAP = {
    "model.diffusion_model.time_embed.0.weight": "time_embedding.linear_1.weight",
    "model.diffusion_model.time_embed.0.bias": "time_embedding.linear_1.bias",
    "model.diffusion_model.time_embed.2.weight": "time_embedding.linear_2.weight",
    "model.diffusion_model.time_embed.2.bias": "time_embedding.linear_2.bias",
    "model.diffusion_model.label_emb.0.0.weight": "add_embedding.linear_1.weight",
    "model.diffusion_model.label_emb.0.0.bias": "add_embedding.linear_1.bias",
    "model.diffusion_model.label_emb.0.2.weight": "add_embedding.linear_2.weight",
    "model.diffusion_model.label_emb.0.2.bias": "add_embedding.linear_2.bias",
    "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
    "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
    "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
    "model.diffusion_model.out.2.weight": "conv_out.weight",
    "model.diffusion_model.out.2.bias": "conv_out.bias",
    
    # Downsamplers
    "model.diffusion_model.input_blocks.3.0.op.weight": "down_blocks.0.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias": "down_blocks.0.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight": "down_blocks.1.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias": "down_blocks.1.downsamplers.0.conv.bias",
    
    # Upsamplers
    "model.diffusion_model.output_blocks.2.2.conv.weight": "up_blocks.0.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.2.2.conv.bias": "up_blocks.0.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "up_blocks.1.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias": "up_blocks.1.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "up_blocks.2.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias": "up_blocks.2.upsamplers.0.conv.bias",
}

# FLUX ComfyUI -> Diffusers remapping (ComfyUI convention)
_COMFY_FLUX_KEY_MAP = {
    # Double blocks
    "double_blocks.{i}.img_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.to_q.weight", 0, 3),
        ("transformer_blocks.{i}.attn.to_k.weight", 1, 3),
        ("transformer_blocks.{i}.attn.to_v.weight", 2, 3),
    ],
    "double_blocks.{i}.img_attn.qkv.bias": [
        ("transformer_blocks.{i}.attn.to_q.bias", 0, 3),
        ("transformer_blocks.{i}.attn.to_k.bias", 1, 3),
        ("transformer_blocks.{i}.attn.to_v.bias", 2, 3),
    ],
    "double_blocks.{i}.txt_attn.qkv.weight": [
        ("transformer_blocks.{i}.attn.add_q_proj.weight", 0, 3),
        ("transformer_blocks.{i}.attn.add_k_proj.weight", 1, 3),
        ("transformer_blocks.{i}.attn.add_v_proj.weight", 2, 3),
    ],
    "double_blocks.{i}.txt_attn.qkv.bias": [
        ("transformer_blocks.{i}.attn.add_q_proj.bias", 0, 3),
        ("transformer_blocks.{i}.attn.add_k_proj.bias", 1, 3),
        ("transformer_blocks.{i}.attn.add_v_proj.bias", 2, 3),
    ],
    "double_blocks.{i}.img_attn.proj.weight": "transformer_blocks.{i}.attn.to_out.0.weight",
    "double_blocks.{i}.img_attn.proj.bias": "transformer_blocks.{i}.attn.to_out.0.bias",
    "double_blocks.{i}.txt_attn.proj.weight": "transformer_blocks.{i}.attn.to_add_out.weight",
    "double_blocks.{i}.txt_attn.proj.bias": "transformer_blocks.{i}.attn.to_add_out.bias",
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
    "single_blocks.{i}.linear1.bias": "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.bias",
    "single_blocks.{i}.linear2.weight": "single_transformer_blocks.{i}.attn.to_out.weight",
    "single_blocks.{i}.linear2.bias": "single_transformer_blocks.{i}.attn.to_out.bias",
    "single_blocks.{i}.modulation.lin.weight": "single_transformer_blocks.{i}.norm.linear.weight",
    "single_blocks.{i}.modulation.lin.bias": "single_transformer_blocks.{i}.norm.linear.bias",
    "single_blocks.{i}.norm.key_norm.scale": "single_transformer_blocks.{i}.attn.norm_k.weight",
    "single_blocks.{i}.norm.query_norm.scale": "single_transformer_blocks.{i}.attn.norm_q.weight",
    
    # Top-level Modulations & Time
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "time_in.in_layer.weight": "time_text_embed.timestep_embedder.linear_1.weight",
    "time_in.in_layer.bias": "time_text_embed.timestep_embedder.linear_1.bias",
    "time_in.out_layer.weight": "time_text_embed.timestep_embedder.linear_2.weight",
    "time_in.out_layer.bias": "time_text_embed.timestep_embedder.linear_2.bias",
    "guidance_in.in_layer.weight": "time_text_embed.guidance_embedder.linear_1.weight",
    "guidance_in.in_layer.bias": "time_text_embed.guidance_embedder.linear_1.bias",
    "guidance_in.out_layer.weight": "time_text_embed.guidance_embedder.linear_2.weight",
    "guidance_in.out_layer.bias": "time_text_embed.guidance_embedder.linear_2.bias",
    "vector_in.in_layer.weight": "time_text_embed.text_embedder.linear_1.weight",
    "vector_in.in_layer.bias": "time_text_embed.text_embedder.linear_1.bias",
    "vector_in.out_layer.weight": "time_text_embed.text_embedder.linear_2.weight",
    "vector_in.out_layer.bias": "time_text_embed.text_embedder.linear_2.bias",
    "txt_in.weight": "context_embedder.weight",
    "txt_in.bias": "context_embedder.bias",
    "img_in.weight": "x_embedder.weight",
    "img_in.bias": "x_embedder.bias",
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
}

# T5 encoder remapping (city96 t5encoder GGUF convention, arch=t5encoder)
_T5_KEY_MAP = {
    # Top-level (no block index)
    "token_embd.weight":              "shared.weight",
    "enc.output_norm.weight":         "encoder.final_layer_norm.weight",
    # Per-block attention
    "enc.blk.{i}.attn_q.weight":     "encoder.block.{i}.layer.0.SelfAttention.q.weight",
    "enc.blk.{i}.attn_k.weight":     "encoder.block.{i}.layer.0.SelfAttention.k.weight",
    "enc.blk.{i}.attn_v.weight":     "encoder.block.{i}.layer.0.SelfAttention.v.weight",
    "enc.blk.{i}.attn_o.weight":     "encoder.block.{i}.layer.0.SelfAttention.o.weight",
    "enc.blk.{i}.attn_rel_b.weight": "encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight",
    "enc.blk.{i}.attn_norm.weight":  "encoder.block.{i}.layer.0.layer_norm.weight",
    # Per-block FFN (T5 v1.1 gated activation: gate=wi_0, up=wi_1)
    "enc.blk.{i}.ffn_gate.weight":   "encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight",
    "enc.blk.{i}.ffn_up.weight":     "encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight",
    "enc.blk.{i}.ffn_down.weight":   "encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
    "enc.blk.{i}.ffn_norm.weight":   "encoder.block.{i}.layer.1.layer_norm.weight",
}

def _build_remap_fn(gguf_keys: List[str], arch: str = "unknown"):
    # Check if T5 encoder format (city96 convention, arch=t5encoder)
    if arch == "t5encoder" or any(k.startswith("enc.blk.") for k in gguf_keys):
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "enc" and parts[1] == "blk":
                try: max_i = max(max_i, int(parts[2]))
                except ValueError: pass
        remap = {}
        for tmpl_src, tmpl_dst in _T5_KEY_MAP.items():
            if "{i}" not in tmpl_src:
                remap[tmpl_src] = [(tmpl_dst, None)]
            else:
                for i in range(max_i + 1):
                    remap[tmpl_src.replace("{i}", str(i))] = [(tmpl_dst.replace("{i}", str(i)), None)]
        def _remap(name: str):
            return remap.get(name, [(name, None)])
        logger.info("[GGUFSeeker] T5 encoder key naming detected — remapping to Diffusers convention.")
        return _remap

    # Check if llama.cpp format
    if any(k.startswith("blk.") or k.startswith("single_blk.") for k in gguf_keys):
        max_i = 0
        for k in gguf_keys:
            parts = k.split(".")
            if parts[0] in ("blk", "single_blk") and len(parts) > 1:
                try: max_i = max(max_i, int(parts[1]))
                except ValueError: pass
        
        remap = {}
        for tmpl_src, tmpl_dst in _LLAMA_KEY_MAP.items():
            for i in range(max_i + 1):
                remap[tmpl_src.replace("{i}", str(i))] = [(tmpl_dst.replace("{i}", str(i)), None)]
                
        def _remap(name: str):
            return remap.get(name, [(name, None)])
        logger.info("[GGUFSeeker] llama.cpp key naming detected — remapping to Diffusers convention.")
        return _remap
        
    
    # Check if SDXL / SD 1.5 ComfyUI format
    elif any(k.startswith("model.diffusion_model.") for k in gguf_keys):
        is_sdxl = any("label_emb." in k for k in gguf_keys) or any("transformer_blocks.9." in k for k in gguf_keys)
        
        if not is_sdxl:
            logger.info("[GGUFSeeker] SD 1.5 key naming detected — remapping to Diffusers convention.")
            remap = _COMFY_SD15_BASE_MAP.copy()
            for ldm_pref, diff_pref, num_t, is_resnet in _SD15_BLOCKS:
                template = _SDXL_RESNET_TEMPLATE if is_resnet else _SDXL_ATTN_TEMPLATE
                for k, v in template.items():
                    if num_t > 0:
                        for j in range(num_t):
                            tk = k.format(j=j)
                            tv = v.format(j=j)
                            remap[f"model.diffusion_model.{ldm_pref}.{tk}"] = f"{diff_pref}.{tv}"
                    else:
                        remap[f"model.diffusion_model.{ldm_pref}.{k}"] = f"{diff_pref}.{v}"

            def _remap(name: str):
                if name in remap:
                    return [(remap[name], None)]
                return [(name, None)]
            return _remap

        logger.info("[GGUFSeeker] SDXL key naming detected — remapping to Diffusers convention.")
        remap = _COMFY_SDXL_BASE_MAP.copy()
        
        for ldm_pref, diff_pref, num_t, is_resnet in _SDXL_BLOCKS:
            if is_resnet:
                for k_src, k_dst in _SDXL_RESNET_TEMPLATE.items():
                    remap[f"model.diffusion_model.{ldm_pref}.{k_src}"] = f"{diff_pref}.{k_dst}"
            else:
                for k_src, k_dst in _SDXL_ATTN_TEMPLATE.items():
                    if "{j}" in k_src:
                        for j in range(num_t):
                            remap[f"model.diffusion_model.{ldm_pref}.{k_src.replace('{j}', str(j))}"] = f"{diff_pref}.{k_dst.replace('{j}', str(j))}"
                    else:
                        remap[f"model.diffusion_model.{ldm_pref}.{k_src}"] = f"{diff_pref}.{k_dst}"

        def _remap(name: str):
            val = remap.get(name)
            if val is not None:
                return [(val, None)]
            return [(name, None)]
        
        logger.info("[GGUFSeeker] SDXL ComfyUI key naming detected — remapping to Diffusers convention (condensed).")
        return _remap

    # Check if FLUX ComfyUI format
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

    # Check if Z-Image format
    elif any(k.startswith("context_refiner.") for k in gguf_keys) or any(k.startswith("noise_refiner.") for k in gguf_keys):
        def _remap(name: str):
            if name.startswith("x_embedder."):
                return [(name.replace("x_embedder.", "all_x_embedder.2-1."), None)]
            if name.startswith("final_layer."):
                return [(name.replace("final_layer.", "all_final_layer.2-1."), None)]
            return [(name, None)]
        logger.info("[GGUFSeeker] Z-Image key naming detected — remapping to Diffusers convention.")
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
