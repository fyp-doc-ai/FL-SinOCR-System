"""Partition SinOCR dataset across clients using Dirichlet distribution.

Usage:
    python partition_scripts/partition_by_dirichlet.py --config configs/base_config.yaml
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partition_scripts.partition_utils import (
    build_char_label_map,
    load_dataset_csv,
    save_partition,
    save_partition_summary,
)


def dirichlet_partition(
    image_paths: list,
    texts: list,
    num_clients: int,
    alpha: float,
    min_samples: int,
    seed: int,
) -> dict:
    """Partition data indices using Dirichlet distribution over label classes.

    Groups samples by the first character of their text label, then
    distributes each group across clients according to Dir(alpha).
    """
    rng = np.random.default_rng(seed)
    label_map = build_char_label_map(texts)

    class_to_indices = defaultdict(list)
    for idx, cls_id in label_map.items():
        class_to_indices[cls_id].append(idx)

    client_indices = defaultdict(list)

    for cls_id, indices in class_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        splits = (np.cumsum(proportions) * len(indices)).astype(int)
        splits = np.clip(splits, 0, len(indices))

        parts = np.split(indices, splits[:-1])
        for client_id, part in enumerate(parts):
            client_indices[client_id].extend(part.tolist())

    # Redistribute if any client is below min_samples
    for cid in range(num_clients):
        if len(client_indices[cid]) < min_samples:
            all_other = []
            for oid in range(num_clients):
                if oid != cid and len(client_indices[oid]) > min_samples * 2:
                    donate = client_indices[oid][:min_samples]
                    client_indices[oid] = client_indices[oid][min_samples:]
                    all_other.extend(donate)
                    if len(all_other) >= min_samples:
                        break
            client_indices[cid].extend(all_other[:min_samples - len(client_indices[cid])])

    return dict(client_indices)


def main():
    parser = argparse.ArgumentParser(description="Dirichlet dataset partitioning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    data_dir = cfg.data.handwritten_train
    partition_dir = cfg.data.partition_dir
    num_clients = cfg.partition.num_clients
    alpha = cfg.partition.alpha
    min_samples = cfg.partition.min_samples_per_client
    seed = cfg.seed

    print(f"Loading dataset from {data_dir}...")
    image_paths, texts = load_dataset_csv(data_dir)
    print(f"  Total samples: {len(image_paths)}")

    print(f"Partitioning into {num_clients} clients (alpha={alpha})...")
    client_indices = dirichlet_partition(
        image_paths, texts, num_clients, alpha, min_samples, seed
    )

    summary = {
        "method": "dirichlet",
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "source": str(data_dir),
        "clients": {},
    }

    for client_id, indices in client_indices.items():
        c_images = [image_paths[i] for i in indices]
        c_texts = [texts[i] for i in indices]
        metadata = {
            "client_id": client_id,
            "partition_type": "dirichlet",
            "alpha": alpha,
        }
        save_partition(partition_dir, client_id, c_images, c_texts, metadata)
        summary["clients"][str(client_id)] = {"num_samples": len(indices)}
        print(f"  Client {client_id}: {len(indices)} samples")

    save_partition_summary(partition_dir, summary)
    print(f"Partition summary saved to {partition_dir}/partition_summary.json")


if __name__ == "__main__":
    main()
