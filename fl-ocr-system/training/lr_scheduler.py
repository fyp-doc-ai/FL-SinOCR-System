"""Learning rate schedulers for local training."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
) -> LambdaLR:
    """Cosine annealing with optional linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
) -> LambdaLR:
    """Linear decay with optional linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule(optimizer: Optimizer) -> LambdaLR:
    """Constant learning rate (no decay)."""
    return LambdaLR(optimizer, lambda _: 1.0)
