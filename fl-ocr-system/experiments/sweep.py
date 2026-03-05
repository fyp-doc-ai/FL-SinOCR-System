"""Hyperparameter sweep runner for FL experiments.

Generates experiment configs from a sweep definition and runs them sequentially.

Usage:
    python experiments/sweep.py --base-config configs/base_config.yaml
"""

import argparse
import copy
import itertools
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


SWEEP_GRID = {
    "fl.algorithm": ["fedavg", "scaffold", "fedopt"],
    "peft.method": ["none", "lora", "encoder_only"],
    "partition.alpha": [0.1, 0.5, 1.0],
}


def generate_sweep_configs(base_config_path: str, output_dir: str) -> list:
    """Generate all config combinations from the sweep grid."""
    base_cfg = OmegaConf.load(base_config_path)

    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())

    configs = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)
        name_parts = []

        for key, val in zip(keys, combo):
            OmegaConf.update(cfg, key, val, merge=True)
            short_key = key.split(".")[-1]
            name_parts.append(f"{short_key}={val}")

        experiment_name = "_".join(name_parts)
        OmegaConf.update(cfg, "logging.experiment_name", experiment_name, merge=True)

        config_path = os.path.join(output_dir, f"{experiment_name}.yaml")
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(cfg, config_path)
        configs.append(config_path)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="configs/sweep")
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs")
    args = parser.parse_args()

    configs = generate_sweep_configs(args.base_config, args.output_dir)
    print(f"Generated {len(configs)} sweep configurations")

    if args.dry_run:
        for c in configs:
            print(f"  {c}")
        return

    for i, config_path in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Sweep {i+1}/{len(configs)}: {config_path}")
        print(f"{'='*60}")
        subprocess.run(
            [sys.executable, "experiments/run_experiment.py", "--config", config_path],
            check=False,
        )


if __name__ == "__main__":
    main()
