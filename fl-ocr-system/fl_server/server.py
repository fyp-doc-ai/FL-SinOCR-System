"""Flower FL server with config-driven aggregation strategy selection.

Wraps the custom aggregators (FedAvg, SCAFFOLD, FedOPT) into a Flower
Strategy so they can be used with flwr.simulation.start_simulation().
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from omegaconf import DictConfig

from fl_server.aggregators import AGGREGATOR_REGISTRY
from fl_server.aggregators.base import AggregatorInterface
from fl_server.aggregators.fedopt import FedOptAggregator
from fl_server.aggregators.scaffold import ScaffoldAggregator


def create_aggregator(cfg: DictConfig, num_model_params: int = 0) -> AggregatorInterface:
    """Factory function to instantiate the configured aggregator."""
    algorithm = cfg.fl.algorithm.lower()

    if algorithm not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )

    if algorithm == "fedavg":
        return AGGREGATOR_REGISTRY[algorithm]()

    elif algorithm == "scaffold":
        return ScaffoldAggregator(
            num_model_params=num_model_params,
            server_lr=cfg.scaffold.server_lr,
        )

    elif algorithm == "fedopt":
        return FedOptAggregator(
            server_lr=cfg.server_optimizer.lr,
            optimizer_type=cfg.server_optimizer.type,
            beta1=cfg.server_optimizer.beta1,
            beta2=cfg.server_optimizer.beta2,
            epsilon=cfg.server_optimizer.epsilon,
        )

    raise ValueError(f"Unhandled algorithm: {algorithm}")


class FLStrategy(Strategy):
    """Flower Strategy wrapper around our custom aggregators.

    Handles client selection, parameter distribution, aggregation,
    and evaluation orchestration.
    """

    def __init__(
        self,
        aggregator: AggregatorInterface,
        initial_parameters: Parameters,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.3,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Any] = None,
        on_fit_config_fn: Optional[Any] = None,
    ):
        self.aggregator = aggregator
        self._initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.current_parameters = initial_parameters

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self._initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Select clients and prepare FitIns with current parameters."""
        sample_size = max(
            self.min_fit_clients,
            int(client_manager.num_available() * self.fraction_fit),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )

        config: Dict[str, Scalar] = {"server_round": server_round}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)

        # For SCAFFOLD, send global control variate info
        if isinstance(self.aggregator, ScaffoldAggregator):
            gc = self.aggregator.get_global_control()
            if gc is not None:
                config["has_global_control"] = True

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Delegate aggregation to the configured aggregator."""
        params, metrics = self.aggregator.aggregate_fit(
            server_round, results, failures
        )
        if params is not None:
            self.current_parameters = params
        return params, metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Any]]:
        """Select clients for evaluation."""
        if self.fraction_evaluate <= 0:
            return []

        from flwr.common import EvaluateIns

        sample_size = max(
            self.min_evaluate_clients,
            int(client_manager.num_available() * self.fraction_evaluate),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
        )

        config: Dict[str, Scalar] = {"server_round": server_round}
        eval_ins = EvaluateIns(parameters, config)
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients."""
        if not results:
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(
            res.loss * res.num_examples for _, res in results
        ) / total_examples

        aggregated_metrics: Dict[str, Scalar] = {}
        metric_keys = set()
        for _, res in results:
            metric_keys.update(res.metrics.keys())

        for key in metric_keys:
            vals = []
            weights = []
            for _, res in results:
                if key in res.metrics:
                    vals.append(float(res.metrics[key]))
                    weights.append(res.num_examples)
            if vals:
                aggregated_metrics[key] = float(
                    np.average(vals, weights=weights)
                )

        return weighted_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (if evaluate_fn is provided)."""
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
