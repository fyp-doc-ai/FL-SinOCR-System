"""LoRA (Low-Rank Adaptation) for TrOCR.

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

Injects low-rank matrices into attention layers. Only LoRA parameters
are trainable, drastically reducing communication in FL.
"""

from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import VisionEncoderDecoderModel


def apply_lora(
    model: VisionEncoderDecoderModel,
    cfg: DictConfig,
) -> VisionEncoderDecoderModel:
    """Apply LoRA to the TrOCR model using HuggingFace PEFT.

    LoRA is applied to the encoder's attention projection layers by default.
    Only the low-rank matrices A and B are trainable.
    """
    lora_cfg = cfg.peft.lora

    target_modules = list(lora_cfg.target_modules)

    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
