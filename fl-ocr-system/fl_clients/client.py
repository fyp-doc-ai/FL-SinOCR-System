"""Flower FL client for TrOCR-based Sinhala OCR.

Supports full fine-tuning and PEFT-aware parameter exchange,
where only trainable parameters are communicated to/from the server.
"""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

from fl_clients.client_utils import create_client_dataloader, load_client_data
from models.model_utils import (
    compute_parameter_bytes,
    get_parameters_as_ndarrays,
    set_parameters_from_ndarrays,
)
from training.trainer import LocalTrainer


class TrOCRFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapping TrOCR for federated training.

    Only trainable parameters (those with requires_grad=True) are
    exchanged with the server, enabling communication savings when
    PEFT methods freeze most of the model.
    """

    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        client_id: int,
        device: str = "cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.client_id = client_id
        self.device = device

        self.trainer = LocalTrainer(
            model=model,
            device=device,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            optimizer_name=cfg.training.optimizer,
            max_grad_norm=cfg.training.max_grad_norm,
        )

        # For SCAFFOLD: local control variate
        self.local_control = None
        self.global_control = None

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return only trainable parameters as NumPy arrays."""
        return get_parameters_as_ndarrays(self.model, trainable_only=True)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load trainable parameters from server into the model."""
        set_parameters_from_ndarrays(self.model, parameters, trainable_only=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train locally and return updated parameters."""
        self.set_parameters(parameters)

        local_epochs = self.cfg.training.local_epochs
        avg_loss = self.trainer.train(self.train_loader, epochs=local_epochs)

        updated_params = self.get_parameters(config={})

        metrics: Dict[str, Scalar] = {
            "train_loss": float(avg_loss),
            "client_id": self.client_id,
            "bytes_uploaded": compute_parameter_bytes(self.model, trainable_only=True),
        }

        num_examples = len(self.train_loader.dataset)

        # For SCAFFOLD: append delta_c to the parameters
        if self.cfg.fl.algorithm == "scaffold":
            delta_c = self._compute_scaffold_delta_c(parameters)
            updated_params = updated_params + delta_c

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local validation data."""
        self.set_parameters(parameters)

        loss, eval_metrics = self.trainer.evaluate(self.val_loader)
        num_examples = len(self.val_loader.dataset)

        metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
        }
        for k, v in eval_metrics.items():
            metrics[k] = float(v)

        return float(loss), num_examples, metrics

    def _compute_scaffold_delta_c(
        self, old_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Compute SCAFFOLD control variate delta for this client.

        delta_c_k = c_k_new - c_k_old
        c_k_new = c_k_old - c + (1 / (K * lr)) * (old_params - new_params)
        """
        new_params = get_parameters_as_ndarrays(self.model, trainable_only=True)
        lr = self.cfg.training.learning_rate
        K = self.cfg.training.local_epochs

        if self.local_control is None:
            self.local_control = [np.zeros_like(p) for p in new_params]
        if self.global_control is None:
            self.global_control = [np.zeros_like(p) for p in new_params]

        delta_c = []
        for i in range(len(new_params)):
            c_new = (
                self.local_control[i]
                - self.global_control[i]
                + (old_params[i] - new_params[i]) / (K * lr)
            )
            dc = c_new - self.local_control[i]
            self.local_control[i] = c_new
            delta_c.append(dc)

        return delta_c


def create_client_fn(
    model_factory,
    processor,
    cfg: DictConfig,
    device: str,
):
    """Return a client_fn for use with flwr.simulation.start_simulation().

    model_factory: callable that returns a fresh copy of the model
    """

    def client_fn(cid: str) -> TrOCRFlowerClient:
        client_id = int(cid)

        image_paths, texts = load_client_data(cfg.data.partition_dir, client_id)

        # Use 80/20 train/val split
        split = int(len(image_paths) * 0.8)
        train_images, val_images = image_paths[:split], image_paths[split:]
        train_texts, val_texts = texts[:split], texts[split:]

        train_loader = create_client_dataloader(
            train_images, train_texts, processor,
            batch_size=cfg.training.batch_size,
            max_length=cfg.model.max_length,
            shuffle=True,
        )
        val_loader = create_client_dataloader(
            val_images, val_texts, processor,
            batch_size=cfg.training.batch_size,
            max_length=cfg.model.max_length,
            shuffle=False,
        )

        model = model_factory()
        model.to(device)

        return TrOCRFlowerClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            client_id=client_id,
            device=device,
        )

    return client_fn
