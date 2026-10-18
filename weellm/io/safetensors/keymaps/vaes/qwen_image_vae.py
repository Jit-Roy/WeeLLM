"""
weellm/io/safetensors/keymaps/vaes/qwen_image_vae.py
Safetensors keymap for the original Wan/Qwen VAE checkpoint format.

The original format (as distributed on HuggingFace under names like
"qwen_image-vae.safetensors") uses a flat module hierarchy that differs
from the Diffusers AutoencoderKLQwenImage re-implementation:

  Original                          Diffusers
  ────────────────────────────────  ──────────────────────────────────
  conv1.*                           quant_conv.*
  conv2.*                           post_quant_conv.*
  encoder.conv1.*                   encoder.conv_in.*
  decoder.conv1.*                   decoder.conv_in.*
  encoder.head.0.gamma              encoder.norm_out.weight
  decoder.head.0.gamma              decoder.norm_out.weight
  encoder.head.2.*                  encoder.conv_out.*
  decoder.head.2.*                  decoder.conv_out.*
  encoder.middle.N.*                encoder.mid_block.resnets/attentions.N.*
  decoder.middle.N.*                decoder.mid_block.resnets/attentions.N.*
  encoder.downsamples.N.*           encoder.down_blocks.N.*
  decoder.upsamples.N.*             decoder.up_blocks.N.*
  *.gamma                           *.weight   (RMS norm scale)
  *.beta                            *.bias

Detection: file contains keys starting with "conv1." and "conv2." at the
top level AND "decoder.upsamples." — this is unique to the original Qwen VAE.
"""

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Mid-block layout: alternating resnet / attention blocks
# Original flat layout:  middle.0 = resnet, middle.1 = attention, middle.2 = resnet
# Diffusers layout:      mid_block.resnets[0], mid_block.attentions[0], mid_block.resnets[1]
# ---------------------------------------------------------------------------
# We detect index parity: even indices are resnets, odd are attentions.

def _map_middle_key(parent: str, idx: int, sub: str) -> str:
    """parent = 'encoder' | 'decoder', idx = flat integer, sub = rest of key."""
    if idx % 2 == 0:
        # Residual block — resnets[idx // 2]
        resnet_idx = idx // 2
        # sub is like "residual.0.gamma" — strip leading "residual."
        if sub.startswith("residual."):
            sub = sub[len("residual."):]
        sub = _map_residual_sub(sub)
        return f"{parent}.mid_block.resnets.{resnet_idx}.{sub}"
    else:
        # Attention block — attentions[(idx - 1) // 2]
        attn_idx = (idx - 1) // 2
        sub = _map_attn_sub(sub)
        return f"{parent}.mid_block.attentions.{attn_idx}.{sub}"



# Residual block sub-key mapping
# Original: 0.gamma, 2.weight/bias, 3.gamma, 6.weight/bias
# Diffusers: norm1.gamma, conv1.weight/bias, norm2.gamma, conv2.weight/bias
_RESBLOCK_IDX_MAP = {
    ("0", "gamma"):  ("norm1", "gamma"),
    ("2", "weight"): ("conv1", "weight"),
    ("2", "bias"):   ("conv1", "bias"),
    ("3", "gamma"):  ("norm2", "gamma"),
    ("6", "weight"): ("conv2", "weight"),
    ("6", "bias"):   ("conv2", "bias"),
}

def _map_residual_sub(sub: str) -> str:
    """Map residual block sub-path, e.g. '0.gamma' -> 'norm1.weight'."""
    m = re.match(r"^(\d+)\.(weight|bias|gamma|beta)$", sub)
    if m:
        lookup = (m.group(1), m.group(2))
        if lookup in _RESBLOCK_IDX_MAP:
            layer, param = _RESBLOCK_IDX_MAP[lookup]
            return f"{layer}.{param}"
    # fallback
    return _gamma_beta(sub)


def _map_attn_sub(sub: str) -> str:
    """Map attention block sub-path."""
    # norm.gamma -> norm.gamma (QwenImageRMS_norm uses .gamma)
    if sub == "norm.gamma":
        return "norm.gamma"
    # proj/to_qkv stay as-is
    return _gamma_beta(sub)


def _gamma_beta(k: str) -> str:
    # QwenImageRMS_norm in diffusers actually kept the name 'gamma'.
    # We only rename beta -> bias if it appears.
    if k.endswith(".beta"):
        return k[:-5] + ".bias"
    return k


def _translate_key(key: str) -> str:
    k = key

    # ── Top-level quant convs ──────────────────────────────────────────────
    if k.startswith("conv1."):
        return k.replace("conv1.", "quant_conv.", 1)
    if k.startswith("conv2."):
        return k.replace("conv2.", "post_quant_conv.", 1)

    # ── conv_in ────────────────────────────────────────────────────────────
    m = re.match(r"^(encoder|decoder)\.conv1\.(.+)$", k)
    if m:
        return f"{m.group(1)}.conv_in.{m.group(2)}"

    # ── norm_out (head.0.gamma) ────────────────────────────────────────────
    m = re.match(r"^(encoder|decoder)\.head\.0\.gamma$", k)
    if m:
        return f"{m.group(1)}.norm_out.gamma"

    # ── conv_out (head.2.*) ────────────────────────────────────────────────
    m = re.match(r"^(encoder|decoder)\.head\.2\.(.+)$", k)
    if m:
        return f"{m.group(1)}.conv_out.{m.group(2)}"

    # ── mid_block (middle.N.*) ─────────────────────────────────────────────
    m = re.match(r"^(encoder|decoder)\.middle\.(\d+)\.(.+)$", k)
    if m:
        parent, idx_str, sub = m.group(1), m.group(2), m.group(3)
        return _map_middle_key(parent, int(idx_str), sub)

    # ── Encoder down_blocks (downsamples.N.*) ─────────────────────────────
    m = re.match(r"^encoder\.downsamples\.(\d+)\.(.+)$", k)
    if m:
        n, sub = m.group(1), m.group(2)
        if sub.startswith("residual."):
            sub = _map_residual_sub(sub[9:])
        elif sub.startswith("shortcut."):
            sub = "conv_" + sub
        else:
            sub = _gamma_beta(sub)
        return f"encoder.down_blocks.{n}.{sub}"

    # ── Decoder up_blocks (upsamples.N.*) ─────────────────────────────────
    m = re.match(r"^decoder\.upsamples\.(\d+)\.(.+)$", k)
    if m:
        n_str, sub = m.group(1), m.group(2)
        n = int(n_str)
        block_idx = n // 4
        rem = n % 4
        if rem < 3:
            if sub.startswith("residual."):
                sub = _map_residual_sub(sub[9:])
            elif sub.startswith("shortcut."):
                sub = "conv_" + sub
            else:
                sub = _gamma_beta(sub)
            return f"decoder.up_blocks.{block_idx}.resnets.{rem}.{sub}"
        else:
            sub = _gamma_beta(sub)
            return f"decoder.up_blocks.{block_idx}.upsamplers.0.{sub}"

    # ── Fallback: gamma/beta ───────────────────────────────────────────────
    return _gamma_beta(k)


class QwenImageVAESafetensorsKeyMap:
    NAME = "qwen-image-vae-original"

    @staticmethod
    def detect(keys: List[str]) -> bool:
        # Original Qwen VAE has top-level conv1/conv2 AND decoder.upsamples
        has_conv12 = any(k.startswith("conv1.") for k in keys) and any(
            k.startswith("conv2.") for k in keys
        )
        has_upsamples = any(k.startswith("decoder.upsamples.") for k in keys)
        return has_conv12 and has_upsamples

    @staticmethod
    def build_remap(keys: List[str]) -> Dict[str, str]:
        remap: Dict[str, str] = {}
        for k in keys:
            remap[k] = _translate_key(k)
        return remap

    @staticmethod
    def postprocess_tensor(diffusers_key: str, tensor: Any, src_key: str) -> Any:
        return tensor
