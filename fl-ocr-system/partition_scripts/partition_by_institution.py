"""Partition SinOCR dataset to simulate institution-level heterogeneity.

Creates clients with different mixes of handwritten and printed data
and varying data volumes to simulate real-world institutional differences.

Usage:
    python partition_scripts/partition_by_institution.py --config configs/base_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partition_scripts.partition_utils import (
    load_dataset_csv,
    save_partition,
    save_partition_summary,
)


def institution_partition(
    hw_images: list,
    hw_texts: list,
    pr_images: list,
    pr_texts: list,
    num_clients: int,
    seed: int,
) -> dict:
    """Create institution-style partitions with different data compositions.

    Strategy:
    - First ~40% of clients: handwritten-only subsets
    - Next ~30% of clients: printed-only subsets
    - Remaining ~30%: mixed handwritten + printed subsets
    Data volume varies per client to simulate uneven institutional data.
    """
    rng = np.random.default_rng(seed)

    hw_indices = np.arange(len(hw_images))
    pr_indices = np.arange(len(pr_images))
    rng.shuffle(hw_indices)
    rng.shuffle(pr_indices)

    n_hw_only = max(1, int(num_clients * 0.4))
    n_pr_only = max(1, int(num_clients * 0.3))
    n_mixed = num_clients - n_hw_only - n_pr_only

    hw_splits = np.array_split(hw_indices, n_hw_only + n_mixed)
    pr_per_client = len(pr_indices) // (n_pr_only + n_mixed)

    client_data = {}
    client_id = 0

    for i in range(n_hw_only):
        idxs = hw_splits[i].tolist()
        client_data[client_id] = {
            "images": [hw_images[j] for j in idxs],
            "texts": [hw_texts[j] for j in idxs],
            "type": "handwritten_only",
        }
        client_id += 1

    pr_offset = 0
    for i in range(n_pr_only):
        end = min(pr_offset + pr_per_client, len(pr_indices))
        idxs = pr_indices[pr_offset:end].tolist()
        client_data[client_id] = {
            "images": [pr_images[j] for j in idxs],
            "texts": [pr_texts[j] for j in idxs],
            "type": "printed_only",
        }
        pr_offset = end
        client_id += 1

    for i in range(n_mixed):
        hw_idxs = hw_splits[n_hw_only + i].tolist()
        pr_end = min(pr_offset + pr_per_client, len(pr_indices))
        pr_idxs = pr_indices[pr_offset:pr_end].tolist()
        pr_offset = pr_end

        images = [hw_images[j] for j in hw_idxs] + [pr_images[j] for j in pr_idxs]
        texts = [hw_texts[j] for j in hw_idxs] + [pr_texts[j] for j in pr_idxs]
        client_data[client_id] = {
            "images": images,
            "texts": texts,
            "type": "mixed",
        }
        client_id += 1

    return client_data


def main():
    parser = argparse.ArgumentParser(description="Institution-level partitioning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    seed = cfg.seed
    partition_dir = cfg.data.partition_dir
    num_clients = cfg.partition.num_clients

    print("Loading handwritten dataset...")
    hw_images, hw_texts = load_dataset_csv(cfg.data.handwritten_train)
    print(f"  Handwritten: {len(hw_images)} samples")

    print("Loading printed dataset...")
    pr_images, pr_texts = load_dataset_csv(cfg.data.printed_train)
    print(f"  Printed: {len(pr_images)} samples")

    print(f"Creating {num_clients} institution-level partitions...")
    client_data = institution_partition(
        hw_images, hw_texts, pr_images, pr_texts, num_clients, seed
    )

    summary = {
        "method": "institution",
        "num_clients": num_clients,
        "seed": seed,
        "clients": {},
    }

    for client_id, data in client_data.items():
        metadata = {
            "client_id": client_id,
            "partition_type": "institution",
            "data_type": data["type"],
        }
        save_partition(partition_dir, client_id, data["images"], data["texts"], metadata)
        summary["clients"][str(client_id)] = {
            "num_samples": len(data["images"]),
            "data_type": data["type"],
        }
        print(f"  Client {client_id} ({data['type']}): {len(data['images'])} samples")

    save_partition_summary(partition_dir, summary)
    print(f"Partition summary saved to {partition_dir}/partition_summary.json")


if __name__ == "__main__":
    main()
