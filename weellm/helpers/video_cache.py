"""
weellm/helpers/video_cache.py
=============================
Generic video inference cache for WeeLLM.

Video models can take hours to run.  This module provides a two-level cache:

  1. **Prompt-embeds cache** — keyed on the prompt text alone.
     Re-running the same prompt with different resolution or step counts skips
     the expensive text-encoder pass (e.g. Gemma-12B for LTX-2.5 takes ~10 min).

  2. **Step-latents + final-latents cache** — keyed on the full generation
     parameters (prompt + height + width + num_frames + steps + seed).
     - A ``callback_on_step_end``-compatible callback writes ``step_NNNN.pt``
       after every N denoising steps, so progress is never fully lost on crash.
     - After a successful full denoising pass the caller should call
       ``save_final(data)``; on the next identical run ``has_final()`` returns
       True and the caller can skip the entire denoising loop.

Directory layout (default root = ``<model_dir>/.weellm_cache``):
::

    .weellm_cache/
      embeds_<prompt_hash>/
        embeds.pt          ← arbitrary dict of tensors (prompt_embeds, masks…)
      run_<run_hash>/
        step_0000.pt       ← latents tensor after step 0
        step_0001.pt
        …
        final.pt           ← final denoised latents dict; presence = safe to skip denoise

Usage example::

    cache = VideoStepCache(
        prompt="A lion at sunset",
        height=512, width=512,
        num_frames=9, steps=4, seed=42,
        cache_root="/path/to/model/.weellm_cache",
    )

    # ── Phase 1: text encoding ───────────────────────────────────────────────
    cached = cache.load_embeds()
    if cached is None:
        embeds = text_encoder(prompt)
        cache.save_embeds({"embeds": embeds, "mask": mask})
    else:
        embeds, mask = cached["embeds"], cached["mask"]

    # ── Phase 2: denoising ───────────────────────────────────────────────────
    if cache.has_final():
        latents = cache.load_final()["latents"]   # skip the loop
    else:
        call_kwargs["callback_on_step_end"] = cache.get_step_callback()
        out = pipe(**call_kwargs)
        latents = out.frames
        cache.save_final({"latents": latents})
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("weellm.cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha1(*parts: str) -> str:
    """Short (16-char) SHA-1 of the concatenated string parts."""
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VideoStepCache:
    """
    Two-level disk cache for video diffusion pipelines.

    Parameters
    ----------
    prompt:
        The text prompt.  Used as the sole key for the embeds cache so the
        same embeds are reused across different resolutions / step counts.
    height, width, num_frames, steps, seed:
        Together with *prompt* these form the run key; any change busts the
        denoising cache (but not the embeds cache).
    save_every:
        Save step latents every N denoising steps (default 1 = every step).
    cache_root:
        Root directory for all cache files.  Defaults to
        ``<cwd>/.weellm_cache``.
    """

    def __init__(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        seed: int,
        save_every: int = 1,
        cache_root: Optional[str] = None,
    ) -> None:
        self.save_every = max(1, int(save_every))
        root = Path(cache_root or os.path.join(os.getcwd(), ".weellm_cache"))

        prompt_hash = _sha1(prompt)
        run_hash    = _sha1(
            prompt,
            str(height), str(width), str(num_frames), str(steps), str(seed),
        )

        self.embeds_dir = root / f"embeds_{prompt_hash}"
        self.run_dir    = root / f"run_{run_hash}"

        self.embeds_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._embeds_path = self.embeds_dir / "embeds.pt"
        self._final_path  = self.run_dir    / "final.pt"

        logger.info(
            "[VideoCache] cache_root=%s | prompt_hash=%s | run_hash=%s",
            root, prompt_hash, run_hash,
        )

    # ── Prompt-embeds cache ───────────────────────────────────────────────────

    def load_embeds(self) -> Optional[Dict[str, Any]]:
        """
        Return the cached prompt-embeds dict, or ``None`` if no cache exists.
        The returned tensors are on CPU; move them to the target device yourself.
        """
        if self._embeds_path.exists():
            logger.info(
                "[VideoCache] Prompt-embeds cache HIT (%s) — skipping text encoder",
                self._embeds_path.name,
            )
            return torch.load(str(self._embeds_path), map_location="cpu", weights_only=False)
        logger.debug("[VideoCache] Prompt-embeds cache MISS")
        return None

    def save_embeds(self, data: Dict[str, Any]) -> None:
        """
        Persist the prompt-embeds dict to disk.
        Tensors are moved to CPU before saving.
        """
        cpu_data = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        }
        torch.save(cpu_data, str(self._embeds_path))
        logger.info("[VideoCache] Prompt-embeds saved → %s", self._embeds_path)

    # ── Final-latents cache ───────────────────────────────────────────────────

    def has_final(self) -> bool:
        """Return True if a completed denoising run is cached on disk."""
        return self._final_path.exists()

    def load_final(self) -> Dict[str, Any]:
        """
        Load the final denoised latents dict from disk.
        Tensors are on CPU; move them to the target device yourself.
        Raises ``FileNotFoundError`` if no cache exists — call ``has_final()`` first.
        """
        if not self._final_path.exists():
            raise FileNotFoundError(f"No final cache at {self._final_path}")
        logger.info(
            "[VideoCache] Final-latents cache HIT (%s) — skipping denoising loop",
            self._final_path.name,
        )
        return torch.load(str(self._final_path), map_location="cpu", weights_only=False)

    def save_final(self, data: Dict[str, Any]) -> None:
        """
        Persist the final denoised latents dict to disk.
        Should be called immediately after the denoising loop succeeds.
        """
        cpu_data = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        }
        torch.save(cpu_data, str(self._final_path))
        logger.info("[VideoCache] Final latents saved → %s", self._final_path)

    # ── Per-step callback ─────────────────────────────────────────────────────

    def get_step_callback(self):
        """
        Return a ``callback_on_step_end`` compatible with all standard diffusers
        pipelines::

            pipe(..., callback_on_step_end=cache.get_step_callback())

        Every ``save_every`` steps the current ``latents`` tensor from
        *callback_kwargs* is checkpointed to ``run_dir/step_NNNN.pt``.
        The callback is a no-op for non-video pipelines that don't produce latents.
        """
        save_every = self.save_every
        run_dir    = self.run_dir

        def _step_callback(pipe, step_index: int, timestep, callback_kwargs: dict):
            if step_index % save_every == 0:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    step_path = run_dir / f"step_{step_index:04d}.pt"
                    torch.save(latents.cpu(), str(step_path))
                    logger.debug(
                        "[VideoCache] Step %4d checkpoint → %s",
                        step_index, step_path.name,
                    )
            return callback_kwargs

        return _step_callback

    # ── encode_prompt interception ────────────────────────────────────────────

    def wrap_pipeline_encode_prompt(self, pipeline, device: str = "cpu") -> bool:
        """
        Monkey-patch *pipeline*.encode_prompt so that:

        * **Cache MISS** – the real ``encode_prompt`` runs as normal, then its
          return value is serialised to ``embeds_<prompt_hash>/embeds.pt``.
        * **Cache HIT** – the heavy text-encoder pass is skipped entirely and
          the cached tensors are returned directly, moved to *device*.

        The patch is applied on the *instance* so it never affects other
        pipeline objects.  The original bound method is captured in the
        closure so ``self`` (the pipeline) is still accessible to it.

        Returns ``True`` if the cache was hit (text encoder skipped),
        ``False`` otherwise (cache was missed; pipeline will encode normally
        and save on the way out).

        Parameters
        ----------
        pipeline:
            A diffusers pipeline object that exposes ``encode_prompt``.
        device:
            Device to move loaded tensors to (e.g. ``"cuda"``).
            Only used on a cache HIT.
        """
        if not hasattr(pipeline, "encode_prompt"):
            logger.debug(
                "[VideoCache] wrap_pipeline_encode_prompt: pipeline has no "
                "encode_prompt method — skipping cache interception."
            )
            return False

        _orig_ep   = pipeline.encode_prompt   # bound method — self is captured
        _cache     = self
        _dev       = device

        cached = _cache.load_embeds()

        if cached is not None:
            # ── HIT: return cached tensors, skip text encoder ─────────────
            def _cached_encode_prompt(*args, **kwargs):  # noqa: ANN202
                logger.info(
                    "[VideoCache] encode_prompt cache HIT — text encoder skipped"
                )
                # Rebuild the return value.  We stored it as
                # {return_0: t0, return_1: t1, …} for tuples,
                # or {return_0: t} for a single tensor.
                result = []
                i = 0
                while f"return_{i}" in cached:
                    val = cached[f"return_{i}"]
                    if isinstance(val, torch.Tensor):
                        val = val.to(_dev)
                    result.append(val)
                    i += 1
                return tuple(result) if len(result) != 1 else result[0]

            pipeline.encode_prompt = _cached_encode_prompt
            return True

        else:
            # ── MISS: wrap to capture + persist the return value ──────────
            def _caching_encode_prompt(*args, **kwargs):  # noqa: ANN202
                result = _orig_ep(*args, **kwargs)

                # Serialise — store tuple returns as indexed keys so we can
                # reconstruct them exactly on the next cache hit.
                if isinstance(result, tuple):
                    save_dict = {
                        f"return_{i}": (v.cpu() if isinstance(v, torch.Tensor) else v)
                        for i, v in enumerate(result)
                    }
                else:
                    save_dict = {
                        "return_0": (
                            result.cpu() if isinstance(result, torch.Tensor) else result
                        )
                    }
                _cache.save_embeds(save_dict)
                return result

            pipeline.encode_prompt = _caching_encode_prompt
            return False

    # ── Utility ───────────────────────────────────────────────────────────────

    def latest_step_latents(self) -> Optional[torch.Tensor]:
        """
        Return the most recently checkpointed step latents, or ``None``.
        Useful for inspecting what the denoiser has produced so far.
        """
        step_files = sorted(self.run_dir.glob("step_*.pt"))
        if not step_files:
            return None
        latest = step_files[-1]
        logger.info("[VideoCache] Latest step checkpoint: %s", latest.name)
        return torch.load(str(latest), map_location="cpu", weights_only=False)

    def clear_run(self) -> None:
        """
        Delete all step + final files for this run (force a fresh denoising pass).
        The embeds cache is kept intact.
        """
        shutil.rmtree(str(self.run_dir), ignore_errors=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[VideoCache] Run cache cleared: %s", self.run_dir)

    def clear_all(self) -> None:
        """Delete both embeds and run caches (full reset)."""
        self.clear_run()
        if self._embeds_path.exists():
            self._embeds_path.unlink()
            logger.info("[VideoCache] Embeds cache cleared: %s", self.embeds_dir)

    @property
    def run_dir_path(self) -> str:
        return str(self.run_dir)

    @property
    def embeds_dir_path(self) -> str:
        return str(self.embeds_dir)
