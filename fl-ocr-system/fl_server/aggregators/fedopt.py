"""FedOPT -- Federated Optimization with server-side adaptive optimizers.

Reddi et al., "Adaptive Federated Optimization", ICLR 2021.

Key idea: treat the aggregated client pseudo-gradient as a gradient
and apply Adam / Adagrad on the server side for stable convergence.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fl_server.aggregators.base import AggregatorInterface, weighted_average


class FedOptAggregator(AggregatorInterface):
    """Server-side adaptive optimization on aggregated pseudo-gradients.

    Supports Adam, Adagrad, and SGD with momentum as the server optimizer.
    """

    def __init__(
        self,
        server_lr: float = 1.0,
        optimizer_type: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.server_lr = server_lr
        self.optimizer_type = optimizer_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.global_params: Optional[List[np.ndarray]] = None
        self.m: Optional[List[np.ndarray]] = None  # first moment
        self.v: Optional[List[np.ndarray]] = None  # second moment
        self.step_count = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        aggregated = weighted_average(results)

        if self.global_params is None:
            self.global_params = [np.copy(p) for p in aggregated]
            self.m = [np.zeros_like(p) for p in aggregated]
            self.v = [np.zeros_like(p) for p in aggregated]
            parameters = ndarrays_to_parameters(aggregated)
        else:
            # pseudo_gradient = aggregated_new - current_global
            pseudo_grad = [
                agg - glob for agg, glob in zip(aggregated, self.global_params)
            ]

            self.step_count += 1
            self.global_params = self._apply_server_optimizer(pseudo_grad)
            parameters = ndarrays_to_parameters(self.global_params)

        metrics: Dict[str, Scalar] = {
            "num_clients": len(results),
            "num_failures": len(failures),
        }
        return parameters, metrics

    def _apply_server_optimizer(
        self, pseudo_grad: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply the server-side optimizer step."""
        updated = []

        for i, g in enumerate(pseudo_grad):
            if self.optimizer_type == "adam":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
                v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
                update = self.server_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            elif self.optimizer_type == "adagrad":
                self.v[i] += g ** 2
                update = self.server_lr * g / (np.sqrt(self.v[i]) + self.epsilon)

            elif self.optimizer_type == "sgd":
                self.m[i] = self.beta1 * self.m[i] + g
                update = self.server_lr * self.m[i]

            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

            updated.append(self.global_params[i] + update)

        return updated

    def get_name(self) -> str:
        return "fedopt"
