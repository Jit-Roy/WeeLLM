"""
ltx25.py -- GGUF key map for LTXV (LTX-2.5) transformer.

Detected by: general.architecture == "ltxv" in GGUF metadata.

The GGUF keys for this architecture match diffusers native keys perfectly, 
so this is just a pass-through keymap to ensure it's explicitly supported 
and avoids falling back to the "unknown architecture" warning.
"""

from typing import Any, Dict, List

class LTX25GGUFKeyMap:
    NAME = "ltxv"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        return arch == "ltxv"

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        mapping = {}
        rename_dict = {
            "patchify_proj": "proj_in",
            "audio_patchify_proj": "audio_proj_in",
            "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
            "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
            "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
            "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
            "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
            "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
            "q_norm": "norm_q",
            "k_norm": "norm_k",
        }
        
        for k in gguf_keys:
            new_key = k
            for rep, ren in rename_dict.items():
                new_key = new_key.replace(rep, ren)
                
            if new_key.startswith("adaln_single."):
                new_key = new_key.replace("adaln_single.", "time_embed.")
            elif new_key.startswith("audio_adaln_single."):
                new_key = new_key.replace("audio_adaln_single.", "audio_time_embed.")
            elif new_key.startswith("prompt_adaln_single."):
                new_key = new_key.replace("prompt_adaln_single.", "prompt_adaln.")
            elif new_key.startswith("audio_prompt_adaln_single."):
                new_key = new_key.replace("audio_prompt_adaln_single.", "audio_prompt_adaln.")
                
            mapping[k] = [(new_key, None)]
            
        return mapping
