"""
base_pipeline.py -- Abstract base class for all WeeLLM streaming pipelines.

Every model implementation (flux2_klein, sd35, sdxl, ...) must subclass
BasePipeline and implement the two abstract methods below.

Adding a new model
------------------
1. Create  weellm/models/<your_model>/pipeline.py
2. Define  class Light<YourModel>Pipeline(BasePipeline)
3. Implement from_pretrained() and generate()
4. Register in weellm/registry.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image


class BasePipeline(ABC):
    """
    Abstract streaming inference pipeline.

    All WeeLLM model pipelines share this interface so that main.py
    can dispatch to any registered model without model-specific code.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "BasePipeline":
        """
        Load and initialise the pipeline from a local model directory.

        Parameters
        ----------
        model_dir : str or Path
            Root directory of the model checkpoint (e.g. ``flux2-klein-4b/``).
        device : str
            Target device (``"cuda"`` or ``"cpu"``).
        dtype : torch.dtype
            Compute dtype (``torch.bfloat16`` recommended).
        **kwargs
            Model-specific keyword arguments (e.g. ``prefetch``, ``force_resplit``).

        Returns
        -------
        BasePipeline
            Fully initialised pipeline ready for generation.
        """
        ...

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Generate an image from a text prompt.

        Parameters
        ----------
        prompt : str
            Natural-language description of the desired image.
        height, width : int
            Output image dimensions in pixels.
        num_inference_steps : int
            Number of denoising steps.
        guidance_scale : float
            Classifier-free guidance strength (1.0 = disabled).
        seed : int, optional
            RNG seed for reproducibility.
        **kwargs
            Model-specific generation kwargs.

        Returns
        -------
        PIL.Image.Image
            The generated image (or a raw tensor when ``output_type="tensor"``).
        """
        ...

    # ------------------------------------------------------------------
    # Optional helpers (may be overridden)
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.__class__.__name__
