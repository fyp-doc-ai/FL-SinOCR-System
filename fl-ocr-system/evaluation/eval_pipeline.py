"""Full evaluation pipeline for federated TrOCR.

Evaluates the global model on the test set and per-client data,
computing OCR metrics (CER, WER) and federated fairness metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from evaluation.metrics import compute_all_metrics
from fl_clients.client_utils import get_num_clients, load_client_data
from models.trocr_wrapper import SinhalaOCRDataset
from partition_scripts.partition_utils import load_dataset_csv


def evaluate_global_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    test_data_dir: str,
    max_length: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate the global model on the full test set."""
    image_paths, texts = load_dataset_csv(test_data_dir)

    dataset = SinhalaOCRDataset(image_paths, texts, processor, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(pixel_values, max_length=max_length)
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(decoded)

    metrics = compute_all_metrics(all_predictions, texts)
    return metrics


def evaluate_per_client(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    partition_dir: str,
    max_length: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate the global model on each client's data separately.

    Returns:
        (aggregated_stats, per_client_metrics)
        aggregated_stats: mean/std/min/max CER across clients
        per_client_metrics: list of metric dicts per client
    """
    num_clients = get_num_clients(partition_dir)
    per_client = []

    model.eval()
    model.to(device)

    for cid in range(num_clients):
        image_paths, texts = load_client_data(partition_dir, cid)
        if not image_paths:
            continue

        dataset = SinhalaOCRDataset(image_paths, texts, processor, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                gen_ids = model.generate(pixel_values, max_length=max_length)
                decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(decoded)

        metrics = compute_all_metrics(all_preds, texts)
        metrics["client_id"] = cid
        per_client.append(metrics)

    cer_values = [m["cer"] for m in per_client]
    aggregated = {
        "mean_cer": float(np.mean(cer_values)) if cer_values else 0.0,
        "std_cer": float(np.std(cer_values)) if cer_values else 0.0,
        "min_cer": float(np.min(cer_values)) if cer_values else 0.0,
        "max_cer": float(np.max(cer_values)) if cer_values else 0.0,
        "worst_client_cer": float(np.max(cer_values)) if cer_values else 0.0,
        "num_clients_evaluated": len(per_client),
    }

    return aggregated, per_client
