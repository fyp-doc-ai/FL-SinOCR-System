"""Shared utilities for dataset partitioning."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def load_dataset_csv(data_dir: str) -> Tuple[List[str], List[str]]:
    """Load image paths and text labels from a dataset directory.

    Supports both data.csv (handwritten) and gt.csv (printed) formats.
    Both use columns: file_name, text
    """
    data_path = Path(data_dir)
    csv_candidates = ["data.csv", "gt.csv"]
    csv_file = None
    for name in csv_candidates:
        if (data_path / name).exists():
            csv_file = data_path / name
            break
    if csv_file is None:
        raise FileNotFoundError(f"No data.csv or gt.csv found in {data_dir}")

    df = pd.read_csv(csv_file)
    assert "file_name" in df.columns and "text" in df.columns

    image_dir = data_path / "images"
    image_paths = []
    texts = []
    for _, row in df.iterrows():
        fname = str(row["file_name"])
        if not fname.endswith(".png"):
            fname += ".png"
        img_path = str(image_dir / fname)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            texts.append(str(row["text"]))

    return image_paths, texts


def save_partition(
    partition_dir: str,
    client_id: int,
    image_paths: List[str],
    texts: List[str],
    metadata: Dict[str, Any],
) -> None:
    """Save a client partition: copy images, write data.csv and metadata.json."""
    client_dir = Path(partition_dir) / f"client_{client_id}"
    images_dir = client_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for src_path, text in zip(image_paths, texts):
        fname = os.path.basename(src_path)
        dst_path = images_dir / fname
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        records.append({"file_name": Path(fname).stem, "text": text})

    pd.DataFrame(records).to_csv(client_dir / "data.csv", index=False)

    metadata["num_samples"] = len(records)
    with open(client_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_partition_summary(partition_dir: str, summary: Dict[str, Any]) -> None:
    """Save a global partition summary JSON."""
    path = Path(partition_dir) / "partition_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def get_unique_chars(texts: List[str]) -> List[str]:
    """Extract unique Sinhala characters from text labels."""
    chars = set()
    for t in texts:
        chars.update(t)
    return sorted(chars)


def build_char_label_map(texts: List[str]) -> Dict[int, int]:
    """Map each sample index to the id of its first character (for Dirichlet grouping)."""
    all_chars = get_unique_chars(texts)
    char_to_id = {c: i for i, c in enumerate(all_chars)}
    label_map = {}
    for idx, text in enumerate(texts):
        if text:
            label_map[idx] = char_to_id[text[0]]
        else:
            label_map[idx] = 0
    return label_map
