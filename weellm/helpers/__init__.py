"""
weellm.helpers
==============
Utility helpers for WeeLLM inference pipelines.

Modules
-------
video_cache
    Two-level disk cache for video diffusion models.
    Saves prompt embeddings (keyed on prompt text) and per-step denoising
    latents (keyed on full generation params) so that long video runs
    survive crashes and repeated runs skip expensive recomputation.
"""
