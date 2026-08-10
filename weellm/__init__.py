"""
WeeLLM — Layer-streaming inference for large diffusion models.

"""

import logging

__version__ = "0.1.0"
__all__ = ["WeePipeline"]

# Add a NullHandler by default so library users don't get "No handlers" warnings.
# Applications that want output should configure their own handlers.
logging.getLogger("weellm").addHandler(logging.NullHandler())

from .pipeline import WeePipeline  # noqa: E402