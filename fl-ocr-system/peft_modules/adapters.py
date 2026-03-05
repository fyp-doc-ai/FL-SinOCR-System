"""Adapter modules for TrOCR.

Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019.

Inserts small bottleneck layers into each transformer block. Only
adapter parameters are trainable.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import VisionEncoderDecoderModel

from models.model_utils import freeze_module


class AdapterLayer(nn.Module):
    """Bottleneck adapter: down-project -> nonlinearity -> up-project + residual."""

    def __init__(self, input_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class AdapterWrappedLayer(nn.Module):
    """Wraps an existing transformer layer and appends an adapter after it."""

    def __init__(self, original_layer: nn.Module, adapter: AdapterLayer):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            hidden_states = self.adapter(hidden_states)
            return (hidden_states,) + outputs[1:]
        return self.adapter(outputs)


def apply_adapters(
    model: VisionEncoderDecoderModel,
    cfg: DictConfig,
) -> VisionEncoderDecoderModel:
    """Insert adapter layers into the TrOCR encoder and freeze everything else.

    Adapters are added after each encoder transformer layer's output.
    """
    bottleneck_dim = cfg.peft.adapter.bottleneck_dim
    dropout = cfg.peft.adapter.adapter_dropout

    freeze_module(model)

    encoder = model.encoder
    hidden_size = encoder.config.hidden_size

    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "layers"):
        layers = encoder.layers
    else:
        raise AttributeError(
            "Cannot locate encoder transformer layers. "
            f"Encoder type: {type(encoder)}"
        )

    for i in range(len(layers)):
        adapter = AdapterLayer(hidden_size, bottleneck_dim, dropout)
        layers[i] = AdapterWrappedLayer(layers[i], adapter)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Adapters] Trainable: {total_trainable:,} / {total_params:,} "
        f"({100 * total_trainable / total_params:.2f}%)"
    )

    return model
