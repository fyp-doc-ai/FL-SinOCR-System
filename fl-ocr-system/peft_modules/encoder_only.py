"""Encoder-only fine-tuning for TrOCR.

Freezes the decoder entirely and only trains the ViT encoder.
This reduces communicated parameters by roughly 50%.
"""

from transformers import VisionEncoderDecoderModel

from models.model_utils import freeze_module, unfreeze_module


def apply_encoder_only(model: VisionEncoderDecoderModel) -> VisionEncoderDecoderModel:
    """Freeze decoder, keep only encoder trainable."""
    freeze_module(model)
    unfreeze_module(model.encoder)

    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(
        f"[Encoder-only] Encoder trainable: {enc_trainable:,}, "
        f"Decoder trainable: {dec_trainable:,}, "
        f"Total: {total:,} ({100 * enc_trainable / total:.2f}%)"
    )

    return model
