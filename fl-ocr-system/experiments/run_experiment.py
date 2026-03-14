"""Main experiment runner for federated TrOCR training.

Reads a YAML config, sets up the model with optional PEFT, configures
the FL aggregation strategy, and launches Flower simulation.

Usage:
    python experiments/run_experiment.py --config configs/fedavg.yaml

Hugging Face: set HF_TOKEN in the environment or in a .env file (in this directory
or project root) to avoid rate limits and use private models. .env is gitignored.
"""

# --- Standard library: CLI, env, paths, types ---
import argparse   # Parse --config and other command-line arguments
import copy       # Deep-copy model for evaluation and client copies
import os         # Environment variables (HF_TOKEN) and path checks
import random     # Client sampling per round
import sys        # Exit on missing partitions
import time       # Measure simulation duration
from pathlib import Path   # Resolve .env path and config paths
from typing import Dict, List, Optional, Tuple   # Type hints for config and aggregates

# Package root directory (fl-ocr-system/); used to find .env and resolve imports
_PKG_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load HF_TOKEN (and similar) from .env so Hugging Face API is authenticated."""
    # Path to .env in package root (fl-ocr-system/.env)
    env_file = _PKG_ROOT / ".env"
    try:
        # Prefer python-dotenv: load from package .env then cwd .env (package overrides)
        from dotenv import load_dotenv
        load_dotenv(env_file, override=True)
        load_dotenv(Path.cwd() / ".env", override=True)
    except ImportError:
        # If python-dotenv not installed, we fall back to manual parsing below
        pass
    # Fallback: read .env manually so it works even if cwd is wrong or dotenv fails
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        if env_file.is_file():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()   # Remove leading/trailing whitespace
                    # Skip empty lines, comments, and lines without "="
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")   # Split on first "="
                        key, value = key.strip(), value.strip().strip('"').strip("'")
                        if key == "HF_TOKEN" or key == "HUGGING_FACE_HUB_TOKEN":
                            os.environ[key] = value   # Make token visible to huggingface_hub
                            break   # One token is enough


_load_env()   # Run at import time so HF_TOKEN is set before any HF calls


def _login_huggingface() -> None:
    """Log in to Hugging Face Hub using HF_TOKEN for higher rate limits and private models."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)   # Authenticate this process with HF
        except Exception:
            pass   # Don’t fail the run if login fails (e.g. network)
    else:
        _env_path = _PKG_ROOT / ".env"
        print(
            "HF_TOKEN not set. To fix: copy .env.example to .env in fl-ocr-system/ and set HF_TOKEN=your_token"
            f" (e.g. create {_env_path})."
        )

# --- Federated learning and ML stack ---
import flwr as fl   # Flower: FL simulation and server/client APIs
import numpy as np  # NumPy arrays for parameter exchange
import torch       # PyTorch model and device handling
from flwr.common import Code, FitIns, FitRes, ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Status
from flwr.common.typing import DisconnectRes, ReconnectIns
from flwr.server.client_proxy import ClientProxy
from omegaconf import DictConfig, OmegaConf   # YAML config load/merge

# Ensure package root is on sys.path so "from evaluation.*" etc. resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.communication_cost import CommunicationTracker   # Track bytes sent per round
from evaluation.eval_pipeline import evaluate_global_model, evaluate_per_client
from fl_clients.client import create_client_fn   # Factory that returns Flower client for a cid
from fl_clients.client_utils import get_num_clients   # Count partition dirs
from fl_server.server import FLStrategy, create_aggregator   # Strategy + FedAvg/SCAFFOLD/FedOpt
from logging_utils.logger import ExperimentLogger   # CSV/TensorBoard/W&B
from models.model_utils import count_parameters, get_parameters_as_ndarrays
from models.trocr_wrapper import TrOCRWrapper   # TrOCR load + processor
from peft_modules.peft_utils import apply_peft   # LoRA / adapter / encoder-only


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and CUDA."""
    torch.manual_seed(seed)   # PyTorch CPU RNG
    np.random.seed(seed)      # NumPy RNG (e.g. client sampling)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # All CUDA devices


def load_config(config_path: str) -> DictConfig:
    """Load experiment YAML and merge with base_config.yaml so defaults are shared."""
    cfg = OmegaConf.load(config_path)   # Load the chosen config (e.g. fedavg.yaml)

    base_path = Path(config_path).parent / "base_config.yaml"
    if base_path.exists() and str(config_path) != str(base_path):
        base_cfg = OmegaConf.load(base_path)   # Load shared defaults
        cfg = OmegaConf.merge(base_cfg, cfg)   # Experiment config overrides base

    return cfg


def build_model(cfg: DictConfig, device: str) -> Tuple:
    """Load TrOCR from config, apply PEFT if configured, return (model, processor)."""
    wrapper = TrOCRWrapper(
        model_name=cfg.model.name,       # e.g. microsoft/trocr-base-handwritten
        max_length=cfg.model.max_length, # Max decoder length for generation
    )
    wrapper.load(device=None)   # Load on CPU first; move to device later

    model = apply_peft(wrapper.model, cfg)   # Optionally add LoRA / adapter / freeze encoder
    processor = wrapper.processor   # Image + text processor for TrOCR

    stats = count_parameters(model)   # total / trainable / frozen counts
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
    """Create a server-side evaluation function called by the FL strategy after aggregation."""

    def evaluate_fn(
        server_round: int, parameters: List[np.ndarray], config: dict
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Only run evaluation every N rounds to save time
        if server_round % cfg.evaluation.eval_every_n_rounds != 0:
            return None

        eval_model = copy.deepcopy(model_template)   # Don’t mutate the shared template
        eval_model.to(device)

        # Apply aggregated parameters (same order as get_parameters_as_ndarrays)
        trainable_params = [p for p in eval_model.parameters() if p.requires_grad]
        for param, values in zip(trainable_params, parameters):
            param.data = torch.from_numpy(values).to(param.device)

        # Generation config for evaluation (aligns with notebook 00_TrOCR_text_fine_tuned_handwritten)
        gen_config = {}
        if hasattr(cfg.model, "num_beams"):
            gen_config["num_beams"] = cfg.model.num_beams
        if hasattr(cfg.model, "length_penalty"):
            gen_config["length_penalty"] = cfg.model.length_penalty
        if hasattr(cfg.model, "early_stopping"):
            gen_config["early_stopping"] = cfg.model.early_stopping
        if hasattr(cfg.model, "no_repeat_ngram_size"):
            gen_config["no_repeat_ngram_size"] = cfg.model.no_repeat_ngram_size

        # Evaluate on global held-out test set (CER/WER)
        test_metrics = evaluate_global_model(
            eval_model,
            processor,
            cfg.data.handwritten_test,
            max_length=cfg.model.max_length,
            device=device,
            gen_config=gen_config or None,
        )

        # Evaluate on each client’s test split and compute mean/worst/std CER
        client_stats, per_client = evaluate_per_client(
            eval_model,
            processor,
            cfg.data.partition_dir,
            max_length=cfg.model.max_length,
            device=device,
            gen_config=gen_config or None,
        )

        # Record how much data was sent this round and cumulative
        comm_entry = comm_tracker.log_round(
            server_round,
            num_clients_fit=cfg.fl.clients_per_round,
        )

        # Build dict of metrics for logging and for Flower’s history
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

        logger.log_round(round_metrics)   # Write to CSV / TensorBoard / W&B
        print(
            f"  Round {server_round}: "
            f"CER={test_metrics.get('cer', 0):.4f}, "
            f"WER={test_metrics.get('wer', 0):.4f}, "
            f"Comm={comm_entry['cumulative_mb']:.1f}MB"
        )

        del eval_model   # Free memory before next round
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Flower expects (loss, metrics_dict); we use CER as loss for strategy callbacks
        loss = test_metrics.get("cer", 1.0)
        return float(loss), {k: float(v) for k, v in round_metrics.items()}

    return evaluate_fn


class _DummyClientProxy(ClientProxy):
    """Minimal ClientProxy for sequential simulation; aggregator only needs client cid + FitRes."""

    def __init__(self, cid: str):
        super().__init__(str(cid))   # Flower needs a string client id

    def get_properties(self, config: dict, timeout: Optional[float] = None):
        raise NotImplementedError   # Not used in our simulation

    def get_parameters(self, config: dict, timeout: Optional[float] = None):
        raise NotImplementedError

    def fit(self, ins: FitIns, timeout: Optional[float] = None):
        raise NotImplementedError   # We call client.fit() directly, not via proxy

    def evaluate(self, ins, timeout: Optional[float] = None):
        raise NotImplementedError

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float] = None, group_id: Optional[int] = None) -> DisconnectRes:
        return DisconnectRes(reason="")   # Satisfy interface; no real reconnect


def run_simulation_sequential(
    client_fn,
    num_clients: int,
    num_rounds: int,
    clients_per_round: int,
    strategy: FLStrategy,
    current_parameters,
    evaluate_fn,
    get_fit_config,
    seed: int,
) -> None:
    """Run FL rounds sequentially without Ray. Use when Ray is not installed (e.g. Python 3.14)."""
    random.seed(seed)
    np.random.seed(seed)

    for server_round in range(1, num_rounds + 1):
        # Sample which clients participate this round
        all_cids = list(range(num_clients))
        selected_cids = random.sample(
            all_cids, min(clients_per_round, num_clients)
        )

        config = get_fit_config(server_round)   # e.g. server_round, local_epochs
        fit_ins = FitIns(current_parameters, config)   # Instructions sent to clients

        results = []
        param_ndarrays = parameters_to_ndarrays(fit_ins.parameters)   # Convert to list of ndarray for client

        for cid in selected_cids:
            client = client_fn(str(cid))   # New client instance with this partition
            parameters, num_examples, metrics = client.fit(param_ndarrays, fit_ins.config)   # Local training
            status = Status(code=Code.OK, message="")
            fit_res = FitRes(
                status=status,
                parameters=ndarrays_to_parameters(parameters),   # Updated weights from client
                num_examples=num_examples,
                metrics=metrics,
            )
            results.append((_DummyClientProxy(str(cid)), fit_res))   # Strategy.aggregate_fit expects (proxy, fit_res)

        # Run FedAvg/SCAFFOLD/FedOpt aggregation and update global parameters
        aggregated_params, _ = strategy.aggregate_fit(server_round, results, [])
        if aggregated_params is not None:
            current_parameters = aggregated_params

        # Run server-side evaluation (global + per-client CER/WER, logging)
        if evaluate_fn is not None:
            eval_ndarrays = parameters_to_ndarrays(current_parameters)
            evaluate_fn(server_round, eval_ndarrays, {})


def main():
    """Entry point: load config, build model, set up FL strategy and clients, run simulation."""
    _login_huggingface()   # Use HF_TOKEN so model download isn’t rate-limited

    parser = argparse.ArgumentParser(description="Run FL experiment")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)   # Load YAML and merge with base_config.yaml
    set_seed(cfg.seed)   # Reproducible client sampling and training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Algorithm: {cfg.fl.algorithm}")
    print(f"PEFT method: {cfg.peft.method}")

    # Load TrOCR and apply PEFT (LoRA / adapter / none)
    model, processor = build_model(cfg, device)

    # CSV, TensorBoard, optional W&B
    logger = ExperimentLogger(cfg)

    # Tracks bytes sent per round (trainable params only)
    comm_tracker = CommunicationTracker(model, trainable_only=True)

    # Serialize initial model weights for Flower (trainable only for PEFT)
    initial_params = get_parameters_as_ndarrays(model, trainable_only=True)
    initial_parameters = ndarrays_to_parameters(initial_params)

    # SCAFFOLD needs number of parameter tensors for control variates
    num_model_params = len(initial_params)

    # FedAvg, SCAFFOLD, or FedOpt from config
    aggregator = create_aggregator(cfg, num_model_params=num_model_params)
    print(f"Aggregator: {aggregator.get_name()}")

    # Called every eval_every_n_rounds: global + per-client CER/WER, comm, logging
    evaluate_fn = create_server_evaluate_fn(
        cfg, processor, model, device, logger, comm_tracker
    )

    # Config sent to clients each round (e.g. server_round, local_epochs)
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

    # Each client gets a deep copy of the model so training doesn’t share state
    def model_factory():
        m = copy.deepcopy(model)
        return m

    # Number of clients = number of partition dirs under partition_dir
    num_clients = get_num_clients(cfg.data.partition_dir)
    if num_clients == 0:
        print(f"ERROR: No client partitions found at {cfg.data.partition_dir}")
        print("Run a partition script first (e.g., partition_by_dirichlet.py)")
        sys.exit(1)

    print(f"Found {num_clients} client partitions")

    # Factory that, given cid, returns a Flower client (NumPyClient) for that partition
    client_fn = create_client_fn(model_factory, processor, cfg, device)

    # Write config and model stats to log dir
    logger.log_config(cfg)
    logger.log_model_info(count_parameters(model))
    comm_summary_before = comm_tracker.get_summary()

    print(f"\nStarting FL simulation: {cfg.fl.num_rounds} rounds, "
          f"{cfg.fl.clients_per_round} clients/round")
    start_time = time.time()

    # Run FL: use Flower’s Ray-based simulation if Ray is installed, else sequential
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0 if device == "cpu" else 0.5},
        )
    except ImportError as e:
        if "ray" in str(e).lower() or "ray" in str(e):
            print("Ray not available (e.g. Python 3.14). Running sequential simulation without Ray...")
            run_simulation_sequential(
                client_fn=client_fn,
                num_clients=num_clients,
                num_rounds=cfg.fl.num_rounds,
                clients_per_round=cfg.fl.clients_per_round,
                strategy=strategy,
                current_parameters=initial_parameters,
                evaluate_fn=evaluate_fn,
                get_fit_config=on_fit_config_fn,
                seed=cfg.seed,
            )
        else:
            raise

    elapsed = time.time() - start_time
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # Write final summary (time, rounds, algorithm, total communication)
    comm_summary = comm_tracker.get_summary()
    logger.log_final_summary({
        "elapsed_seconds": elapsed,
        "total_rounds": cfg.fl.num_rounds,
        "algorithm": cfg.fl.algorithm,
        "peft_method": cfg.peft.method,
        **comm_summary,
    })

    logger.close()   # Flush and close CSV / writers
    print(f"Results saved to: {logger.output_dir}")


if __name__ == "__main__":
    main()   # Run when script is executed directly (not imported)
