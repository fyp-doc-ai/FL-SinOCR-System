"""Local training loop for TrOCR in federated learning.

Handles a single client's local training across multiple epochs,
with configurable optimizer, gradient clipping, and evaluation.
"""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel


class LocalTrainer:
    """Manages local training for an FL client."""

    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        device: str = "cpu",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                trainable_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(self, dataloader: DataLoader, epochs: int = 1) -> float:
        """Run local training for the specified number of epochs.

        Returns the average training loss across all batches.
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        total_steps = 0

        for epoch in range(epochs):
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )

                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on a dataloader, returning loss and generated predictions.

        Returns (avg_loss, metrics_dict). The metrics dict contains 'eval_loss'.
        For OCR metrics (CER/WER), use evaluation/metrics.py separately.
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        metrics = {"eval_loss": avg_loss}

        return avg_loss, metrics
