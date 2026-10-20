from typing import Set, List, Tuple, Any, Callable

class LTX2LoRAKeyMap:
    NAME = "ltx2"

    @staticmethod
    def detect(lora_keys: Set[str]) -> bool:
        """Detect LTX-2.5 specific keys."""
        for k in lora_keys:
            if "patchify_proj" in k or "av_ca_video" in k or "audio_embeddings_connector" in k or "video_embeddings_connector" in k:
                return True
        return False

    @staticmethod
    def build_pairs(lora_keys: Set[str]) -> List[Tuple[str, str, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()
        
        # Determine suffixes based on what's available
        a_suffix = ".lora_A.weight"
        b_suffix = ".lora_B.weight"
        
        for k in lora_keys:
            if a_suffix in k:
                bases.add(k.split(a_suffix)[0])

        for base in bases:
            a_key = f"{base}{a_suffix}"
            b_key = f"{base}{b_suffix}"
            if a_key not in lora_keys or b_key not in lora_keys:
                continue

            base_path = base
            
            is_diff_model = base_path.startswith("diffusion_model.")
            is_text_proj = base_path.startswith("text_embedding_projection.")
            
            if is_diff_model or is_text_proj:
                if is_diff_model:
                    base_path = base_path[len("diffusion_model."):]
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
                        "audio_prompt_adaln_single": "audio_prompt_adaln",
                        "prompt_adaln_single": "prompt_adaln",
                    }
                else:
                    base_path = base_path[len("text_embedding_projection."):]
                    rename_dict = {"aggregate_embed": "text_proj_in"}

                for old_p, new_p in rename_dict.items():
                    base_path = base_path.replace(old_p, new_p)

                if base_path.startswith("adaln_single."):
                    base_path = base_path.replace("adaln_single.", "time_embed.")
                elif base_path.startswith("audio_adaln_single."):
                    base_path = base_path.replace("audio_adaln_single.", "audio_time_embed.")
                    
                if base_path.startswith("audio_embeddings_connector.transformer_1d_blocks."):
                    base_path = base_path.replace("audio_embeddings_connector.transformer_1d_blocks.", "audio_connector.transformer_blocks.")
                    pfx = "connectors."
                elif base_path.startswith("video_embeddings_connector.transformer_1d_blocks."):
                    base_path = base_path.replace("video_embeddings_connector.transformer_1d_blocks.", "video_connector.transformer_blocks.")
                    pfx = "connectors."
                else:
                    pfx = "transformer." if is_diff_model else "connectors."
                base_path = pfx + base_path
            elif base_path.startswith("base_model.model."):
                base_path = base_path[len("base_model.model."):]
                if base_path.startswith("audio_embeddings_connector.transformer_1d_blocks."):
                    base_path = base_path.replace("audio_embeddings_connector.transformer_1d_blocks.", "audio_connector.transformer_blocks.")
                    base_path = "connectors." + base_path
                elif base_path.startswith("video_embeddings_connector.transformer_1d_blocks."):
                    base_path = base_path.replace("video_embeddings_connector.transformer_1d_blocks.", "video_connector.transformer_blocks.")
                    base_path = "connectors." + base_path
                else:
                    base_path = "transformer." + base_path
            elif base_path.startswith("transformer."):
                pass
                
            targets = [(base_path + ".weight", None)]
            pairs.append((a_key, b_key, targets))
            
        return pairs
