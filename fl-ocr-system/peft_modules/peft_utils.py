"""Shared PEFT utilities for applying parameter-efficient fine-tuning to TrOCR."""

from typing import Dict

from omegaconf import DictConfig
from transformers import VisionEncoderDecoderModel

from models.model_utils import count_parameters, freeze_module


def apply_peft(model: VisionEncoderDecoderModel, cfg: DictConfig) -> VisionEncoderDecoderModel:
    """Apply the configured PEFT method to the model.

    Returns the (possibly wrapped) model with only the intended
    parameters set to requires_grad=True.
    """
    method = cfg.peft.method.lower()

    if method == "none":
        return model

    elif method == "lora":
        from peft_modules.lora import apply_lora
        model = apply_lora(model, cfg)

    elif method == "adapter":
        from peft_modules.adapters import apply_adapters
        model = apply_adapters(model, cfg)

    elif method == "encoder_only":
        from peft_modules.encoder_only import apply_encoder_only
        model = apply_encoder_only(model)

    else:
        raise ValueError(f"Unknown PEFT method: {method}")

    stats = count_parameters(model)
    print(f"[PEFT={method}] Parameters: {stats}")

    return model
