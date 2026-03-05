"""Partition SinOCR dataset by visual clustering (writer-level simulation).

Extracts lightweight visual features from each image, clusters them with
K-Means, and assigns each cluster to a simulated FL client.

Usage:
    python partition_scripts/partition_by_clustering.py --config configs/base_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partition_scripts.partition_utils import (
    load_dataset_csv,
    save_partition,
    save_partition_summary,
)


def extract_visual_features(image_paths: list, target_size: tuple = (64, 64)) -> np.ndarray:
    """Extract simple visual features by resizing images to a flat vector.

    For a more sophisticated approach, replace this with TrOCR encoder features.
    """
    features = []
    for path in image_paths:
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize(target_size)
        feat = np.array(img, dtype=np.float32).flatten()
        feat = feat / 255.0
        features.append(feat)
    return np.stack(features)


def cluster_partition(
    image_paths: list,
    texts: list,
    num_clusters: int,
    seed: int,
) -> dict:
    """Cluster images by visual features and return client->indices mapping."""
    print("  Extracting visual features...")
    features = extract_visual_features(image_paths)

    print(f"  Running K-Means with K={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features)

    client_indices = {}
    for cluster_id in range(num_clusters):
        client_indices[cluster_id] = np.where(labels == cluster_id)[0].tolist()

    return client_indices


def main():
    parser = argparse.ArgumentParser(description="Visual clustering partitioning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    data_dir = cfg.data.handwritten_train
    partition_dir = cfg.data.partition_dir
    num_clusters = cfg.partition.num_clusters
    seed = cfg.seed

    print(f"Loading dataset from {data_dir}...")
    image_paths, texts = load_dataset_csv(data_dir)
    print(f"  Total samples: {len(image_paths)}")

    print(f"Clustering into {num_clusters} writer groups...")
    client_indices = cluster_partition(image_paths, texts, num_clusters, seed)

    summary = {
        "method": "clustering",
        "num_clusters": num_clusters,
        "seed": seed,
        "source": str(data_dir),
        "clients": {},
    }

    for client_id, indices in client_indices.items():
        c_images = [image_paths[i] for i in indices]
        c_texts = [texts[i] for i in indices]
        metadata = {
            "client_id": client_id,
            "partition_type": "clustering",
            "cluster_id": client_id,
        }
        save_partition(partition_dir, client_id, c_images, c_texts, metadata)
        summary["clients"][str(client_id)] = {"num_samples": len(indices)}
        print(f"  Client {client_id} (cluster): {len(indices)} samples")

    save_partition_summary(partition_dir, summary)
    print(f"Partition summary saved to {partition_dir}/partition_summary.json")


if __name__ == "__main__":
    main()
