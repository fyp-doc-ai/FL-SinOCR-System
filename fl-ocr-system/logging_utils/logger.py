"""Experiment logging: CSV, TensorBoard, and optional Weights & Biases."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    """Unified logger that writes metrics to CSV, TensorBoard, and optionally W&B."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        experiment_name = cfg.logging.experiment_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(cfg.logging.output_dir) / f"{experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self._csv_headers_written = False

        # TensorBoard
        self.tb_writer = None
        if cfg.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tb_logs"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                print("Warning: tensorboard not available, skipping TB logging")

        # Weights & Biases
        self.wandb_run = None
        if cfg.logging.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=cfg.logging.wandb_project,
                    name=f"{experiment_name}_{timestamp}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
            except ImportError:
                print("Warning: wandb not available, skipping W&B logging")

    def log_round(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for a single FL round."""
        server_round = metrics.get("round", 0)

        # CSV
        if not self._csv_headers_written:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()
            self._csv_headers_written = True
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"fl/{key}", value, server_round)

        # W&B
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=server_round)

    def log_config(self, cfg: DictConfig) -> None:
        """Save the full experiment config to a YAML file."""
        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(cfg, str(config_path))

    def log_model_info(self, param_stats: Dict[str, Any]) -> None:
        """Log model parameter statistics."""
        info_path = self.output_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(param_stats, f, indent=2)

    def log_final_summary(self, summary: Dict[str, Any]) -> None:
        """Log the final experiment summary."""
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("\n--- Experiment Summary ---")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    def close(self) -> None:
        """Clean up all logger resources."""
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
