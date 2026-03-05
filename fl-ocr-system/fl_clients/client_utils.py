"""Utilities for FL client data loading and preprocessing."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor

from models.trocr_wrapper import SinhalaOCRDataset


def load_client_data(
    partition_dir: str,
    client_id: int,
) -> Tuple[List[str], List[str]]:
    """Load image paths and texts for a specific client from its partition folder."""
    client_dir = Path(partition_dir) / f"client_{client_id}"
    csv_path = client_dir / "data.csv"
    images_dir = client_dir / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"No data.csv in {client_dir}")

    df = pd.read_csv(csv_path)
    image_paths = []
    texts = []
    for _, row in df.iterrows():
        fname = str(row["file_name"])
        if not fname.endswith(".png"):
            fname += ".png"
        img_path = str(images_dir / fname)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            texts.append(str(row["text"]))

    return image_paths, texts


def load_client_metadata(partition_dir: str, client_id: int) -> dict:
    """Load client metadata JSON."""
    meta_path = Path(partition_dir) / f"client_{client_id}" / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def create_client_dataloader(
    image_paths: List[str],
    texts: List[str],
    processor: TrOCRProcessor,
    batch_size: int = 8,
    max_length: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for a client's partition."""
    dataset = SinhalaOCRDataset(image_paths, texts, processor, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def get_num_clients(partition_dir: str) -> int:
    """Count the number of client directories in the partition folder."""
    p = Path(partition_dir)
    if not p.exists():
        return 0
    return len([d for d in p.iterdir() if d.is_dir() and d.name.startswith("client_")])
