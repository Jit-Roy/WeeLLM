from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from ..base_streamer import BaseTransformerStreamer

class LTX2ConnectorsStreamer(BaseTransformerStreamer):
    def _get_model_cls(self):
        return LTX2TextConnectors

    def _get_resident_keys(self):
        return [
            "text_proj_in",
            "video_text_proj_in",
            "audio_text_proj_in",
            "video_connector.learnable_registers",
            "video_connector.norm_out",
            "audio_connector.learnable_registers",
            "audio_connector.norm_out"
        ]

    def _get_shard_order(self):
        order = []
        
        # 1. Video Text Projection (1.44 GB)
        if hasattr(self.model, "video_text_proj_in") and self.model.video_text_proj_in is not None:
            order.append(("video_text_proj_in", self.model.video_text_proj_in))
            
        # 2. Audio Text Projection (1.44 GB)
        if hasattr(self.model, "audio_text_proj_in") and self.model.audio_text_proj_in is not None:
            order.append(("audio_text_proj_in", self.model.audio_text_proj_in))
            
        # 3. LTX-2.0 Text Projection
        if hasattr(self.model, "text_proj_in") and self.model.text_proj_in is not None:
            order.append(("text_proj_in", self.model.text_proj_in))
        
        # 8 video blocks
        num_video_layers = self.model.config.video_connector_num_layers
        for i in range(num_video_layers):
            block = self.model.video_connector.transformer_blocks[i]
            order.append((f"video_connector.transformer_blocks.{i}", block))
            
        # 8 audio blocks
        num_audio_layers = self.model.config.audio_connector_num_layers
        for i in range(num_audio_layers):
            block = self.model.audio_connector.transformer_blocks[i]
            order.append((f"audio_connector.transformer_blocks.{i}", block))
            
        return order
        
    def _get_resident_ckpt_keys(self):
        # Exclude projection layers from resident VRAM, they are huge (1.44 GB each).
        # We also exclude transformer blocks which are streamed.
        return [
            k for k in self.seeker.weight_map
            if not k.startswith("video_connector.transformer_blocks.") 
            and not k.startswith("audio_connector.transformer_blocks.")
            and not k.startswith("video_text_proj_in")
            and not k.startswith("audio_text_proj_in")
            and not k.startswith("text_proj_in")
        ]

    def _ckpt_shard_name(self, diffusers_shard_name: str) -> str:
        return diffusers_shard_name

    def _get_layer_keys(self, shard_name: str) -> list[str]:
        return [
            k for k in self.seeker.weight_map
            if k.startswith(f"{shard_name}.")
        ]

    @classmethod
    def from_pretrained(
        cls,
        transformer_dir,
        device="cuda",
        dtype=None,
        prefetch=True,
        prefetch_device=None,
        cache_to_ram=False,
    ):
        from pathlib import Path
        from accelerate import init_empty_weights
        from weellm.seeker import get_seeker
        from weellm.utils import default_dtype
        
        transformer_dir = Path(transformer_dir)
        seeker = get_seeker(transformer_dir, cache_to_ram=cache_to_ram)
        
        config = LTX2TextConnectors.load_config(str(transformer_dir))
        with init_empty_weights(), default_dtype(dtype):
            model = LTX2TextConnectors.from_config(config)
        model.eval()

        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch, prefetch_device=prefetch_device)
        resident_ckpt_keys = streamer._get_resident_ckpt_keys()

        if resident_ckpt_keys:
            raw_sd = seeker.get_tensors(resident_ckpt_keys, device=device, dtype=dtype)
            # Keys in connectors match the checkpoint exactly
            streamer.apply_state_dict(raw_sd, skip_errors=True)
            del raw_sd

        return streamer
