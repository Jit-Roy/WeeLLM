"""GGUF key map for Flux.2 Klein transformers."""

from typing import Any, Dict, List

from weellm.io.ggufs.keymaps.transformers.flux import FluxGGUFKeyMap


class Flux2GGUFKeyMap(FluxGGUFKeyMap):
    NAME = "flux2-klein"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        has_klein_modulation = {
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
        }.issubset(gguf_keys)
        has_flux2_inputs = {
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
        }.issubset(gguf_keys)
        has_flux1_only_inputs = any(
            key.startswith(("guidance_in.", "vector_in."))
            for key in gguf_keys
        )
        return arch in ("flux2", "flux2-klein") or (
            has_klein_modulation
            and has_flux2_inputs
            and not has_flux1_only_inputs
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap = FluxGGUFKeyMap.build_remap(gguf_keys)
        for entries in remap.values():
            for index, (target, slice_info) in enumerate(entries):
                entries[index] = (target.replace("time_text_embed.", "time_guidance_embed."), slice_info)
        return remap
