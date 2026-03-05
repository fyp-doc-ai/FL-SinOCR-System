"""Main experiment runner for federated TrOCR training.

Reads a YAML config, sets up the model with optional PEFT, configures
the FL aggregation strategy, and launches Flower simulation.

Usage:
    python experiments/run_experiment.py --config configs/fedavg.yaml
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.communication_cost import CommunicationTracker
from evaluation.eval_pipeline import evaluate_global_model, evaluate_per_client
from fl_clients.client import create_client_fn
from fl_clients.client_utils import get_num_clients
from fl_server.server import FLStrategy, create_aggregator
from logging_utils.logger import ExperimentLogger
from models.model_utils import count_parameters, get_parameters_as_ndarrays
from models.trocr_wrapper import TrOCRWrapper
from peft_modules.peft_utils import apply_peft


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> DictConfig:
    """Load and merge config with base config."""
    cfg = OmegaConf.load(config_path)

    base_path = Path(config_path).parent / "base_config.yaml"
    if base_path.exists() and str(config_path) != str(base_path):
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)

    return cfg


def build_model(cfg: DictConfig, device: str) -> Tuple:
    """Load TrOCR, apply PEFT, return (model, processor)."""
    wrapper = TrOCRWrapper(
        model_name=cfg.model.name,
        max_length=cfg.model.max_length,
    )
    wrapper.load(device=None)  # load to CPU first

    model = apply_peft(wrapper.model, cfg)
    processor = wrapper.processor

    stats = count_parameters(model)
    print(f"Model loaded: {stats}")

    return model, processor


def create_server_evaluate_fn(
    cfg: DictConfig,
    processor,
    model_template,
    device: str,
    logger: "ExperimentLogger",
    comm_tracker: CommunicationTracker,
):
    """Create a server-side evaluation function for the FL strategy."""

    def evaluate_fn(
        server_round: int, parameters: List[np.ndarray], config: dict
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round % cfg.evaluation.eval_every_n_rounds != 0:
            return None

        eval_model = copy.deepcopy(model_template)
        eval_model.to(device)

        trainable_params = [p for p in eval_model.parameters() if p.requires_grad]
        for param, values in zip(trainable_params, parameters):
            param.data = torch.from_numpy(values).to(param.device)

        # Global test set evaluation
        test_metrics = evaluate_global_model(
            eval_model,
            processor,
            cfg.data.handwritten_test,
            max_length=cfg.model.max_length,
            device=device,
        )

        # Per-client evaluation
        client_stats, per_client = evaluate_per_client(
            eval_model,
            processor,
            cfg.data.partition_dir,
            max_length=cfg.model.max_length,
            device=device,
        )

        # Communication cost
        comm_entry = comm_tracker.log_round(
            server_round,
            num_clients_fit=cfg.fl.clients_per_round,
        )

        # Assemble round metrics
        round_metrics = {
            "round": server_round,
            "global_cer": test_metrics.get("cer", 0),
            "global_wer": test_metrics.get("wer", 0),
            "mean_client_cer": client_stats.get("mean_cer", 0),
            "worst_client_cer": client_stats.get("worst_client_cer", 0),
            "std_client_cer": client_stats.get("std_cer", 0),
            "round_comm_mb": comm_entry["round_total_mb"],
            "cumulative_comm_mb": comm_entry["cumulative_mb"],
        }

        logger.log_round(round_metrics)
        print(
            f"  Round {server_round}: "
            f"CER={test_metrics.get('cer', 0):.4f}, "
            f"WER={test_metrics.get('wer', 0):.4f}, "
            f"Comm={comm_entry['cumulative_mb']:.1f}MB"
        )

        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss = test_metrics.get("cer", 1.0)
        return float(loss), {k: float(v) for k, v in round_metrics.items()}

    return evaluate_fn


def main():
    parser = argparse.ArgumentParser(description="Run FL experiment")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Algorithm: {cfg.fl.algorithm}")
    print(f"PEFT method: {cfg.peft.method}")

    # Build model and processor
    model, processor = build_model(cfg, device)

    # Logger
    logger = ExperimentLogger(cfg)

    # Communication tracker
    comm_tracker = CommunicationTracker(model, trainable_only=True)

    # Get initial parameters (trainable only)
    initial_params = get_parameters_as_ndarrays(model, trainable_only=True)
    initial_parameters = ndarrays_to_parameters(initial_params)

    # Count model params for SCAFFOLD
    num_model_params = len(initial_params)

    # Create aggregator
    aggregator = create_aggregator(cfg, num_model_params=num_model_params)
    print(f"Aggregator: {aggregator.get_name()}")

    # Server-side evaluation
    evaluate_fn = create_server_evaluate_fn(
        cfg, processor, model, device, logger, comm_tracker
    )

    # Build strategy
    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": cfg.training.local_epochs,
        }

    strategy = FLStrategy(
        aggregator=aggregator,
        initial_parameters=initial_parameters,
        fraction_fit=cfg.fl.fraction_fit,
        fraction_evaluate=cfg.fl.fraction_evaluate,
        min_fit_clients=min(2, cfg.fl.clients_per_round),
        min_evaluate_clients=1,
        min_available_clients=cfg.fl.clients_per_round,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
    )

    # Model factory for clients (each client gets a fresh copy)
    def model_factory():
        m = copy.deepcopy(model)
        return m

    # Client function
    num_clients = get_num_clients(cfg.data.partition_dir)
    if num_clients == 0:
        print(f"ERROR: No client partitions found at {cfg.data.partition_dir}")
        print("Run a partition script first (e.g., partition_by_dirichlet.py)")
        sys.exit(1)

    print(f"Found {num_clients} client partitions")

    client_fn = create_client_fn(model_factory, processor, cfg, device)

    # Log experiment start
    logger.log_config(cfg)
    logger.log_model_info(count_parameters(model))
    comm_summary_before = comm_tracker.get_summary()

    print(f"\nStarting FL simulation: {cfg.fl.num_rounds} rounds, "
          f"{cfg.fl.clients_per_round} clients/round")
    start_time = time.time()

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device == "cpu" else 0.5},
    )

    elapsed = time.time() - start_time
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # Final summary
    comm_summary = comm_tracker.get_summary()
    logger.log_final_summary({
        "elapsed_seconds": elapsed,
        "total_rounds": cfg.fl.num_rounds,
        "algorithm": cfg.fl.algorithm,
        "peft_method": cfg.peft.method,
        **comm_summary,
    })

    logger.close()
    print(f"Results saved to: {logger.output_dir}")


if __name__ == "__main__":
    main()
