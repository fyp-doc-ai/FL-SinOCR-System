"""Utility functions for model parameter manipulation in federated learning."""

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
from torch import nn


def get_parameters_as_ndarrays(model: nn.Module, trainable_only: bool = True) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays.

    When trainable_only=True, only parameters with requires_grad are returned.
    This is critical for PEFT where only a small subset of params are communicated.
    """
    if trainable_only:
        return [
            p.cpu().detach().numpy()
            for p in model.parameters()
            if p.requires_grad
        ]
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_parameters_from_ndarrays(
    model: nn.Module,
    parameters: List[np.ndarray],
    trainable_only: bool = True,
) -> None:
    """Load parameters from a list of NumPy arrays into the model.

    When trainable_only=True, only parameters with requires_grad are updated
    (matching the order of get_parameters_as_ndarrays).
    """
    if trainable_only:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = list(model.parameters())

    assert len(parameters) == len(trainable_params), (
        f"Mismatch: received {len(parameters)} param arrays "
        f"but model has {len(trainable_params)} target parameters"
    )

    for param, new_values in zip(trainable_params, parameters):
        param.data = torch.from_numpy(new_values).to(param.device)


def compute_parameter_bytes(model: nn.Module, trainable_only: bool = True) -> int:
    """Compute total bytes for the (trainable) parameters -- float32 = 4 bytes each."""
    if trainable_only:
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        count = sum(p.numel() for p in model.parameters())
    return count * 4  # float32


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return a summary dict of total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100 * trainable / total, 2) if total > 0 else 0,
    }
