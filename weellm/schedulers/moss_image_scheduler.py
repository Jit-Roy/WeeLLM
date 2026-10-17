"""
moss_image_scheduler.py -- FlowMatch Euler Discrete Scheduler for MOSS-Image models.
"""

from typing import Optional, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """
    FlowMatch Euler scheduler using MOSS-Image's training V2, nearest-grid schedule.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        training_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0, **kwargs)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps: int = 50, device: Optional[Union[str, torch.device]] = None, **kwargs):
        if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
            raise ValueError("Steps and shifts must be positive.")
        total = self.config.num_train_timesteps

        def grid(count: int, shift: float):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        # Our sigmas have already been shifted and snapped. Avoid a second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist(), **kwargs)
