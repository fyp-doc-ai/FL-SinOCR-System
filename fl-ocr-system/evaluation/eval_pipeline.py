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


def _generate_kwargs(max_length: int, gen_config: Optional[dict] = None) -> dict:
    """Build kwargs for model.generate() to match notebook (beam search, length penalty).

    Passes params as kwargs so generation uses them instead of deprecated model.config.
    """
    kwargs = {"max_length": max_length}
    if gen_config:
        kwargs["num_beams"] = gen_config.get("num_beams", 4)
        kwargs["length_penalty"] = gen_config.get("length_penalty", 2.0)
        if "early_stopping" in gen_config:
            kwargs["early_stopping"] = gen_config["early_stopping"]
        if "no_repeat_ngram_size" in gen_config:
            kwargs["no_repeat_ngram_size"] = gen_config["no_repeat_ngram_size"]
    return kwargs


def evaluate_global_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    test_data_dir: str,
    max_length: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
    gen_config: Optional[dict] = None,
) -> Dict[str, float]:
    """Evaluate the global model on the full test set.

    gen_config: optional dict with num_beams, length_penalty, early_stopping (aligns with
    notebook 00_TrOCR_text_fine_tuned_handwritten for consistent CER/WER).
    """
    image_paths, texts = load_dataset_csv(test_data_dir)

    dataset = SinhalaOCRDataset(image_paths, texts, processor, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    gen_kwargs = _generate_kwargs(max_length, gen_config)
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(pixel_values, **gen_kwargs)
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
    gen_config: Optional[dict] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate the global model on each client's data separately.

    gen_config: optional dict for model.generate (num_beams, length_penalty, etc.).

    Returns:
        (aggregated_stats, per_client_metrics)
        aggregated_stats: mean/std/min/max CER across clients
        per_client_metrics: list of metric dicts per client
    """
    num_clients = get_num_clients(partition_dir)
    per_client = []
    gen_kwargs = _generate_kwargs(max_length, gen_config)

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
                gen_ids = model.generate(pixel_values, **gen_kwargs)
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
