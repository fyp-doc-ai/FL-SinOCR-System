"""SCAFFOLD -- Stochastic Controlled Averaging for Federated Learning.

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning", ICML 2020.

Key idea: maintain server and client control variates to correct for
client drift caused by heterogeneous data distributions.
"""

import json
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

from fl_server.aggregators.base import AggregatorInterface


class ScaffoldAggregator(AggregatorInterface):
    """SCAFFOLD aggregation with control variate correction.

    The client must send delta_c (control variate update) serialized as JSON
    in fit_res.metrics["delta_c_shapes"] and the flattened delta in
    fit_res.metrics["delta_c_flat"], OR send the delta_c arrays appended
    after the model parameters in the Parameters object.

    For simplicity, this implementation uses the latter approach:
    client sends [model_params..., delta_c_params...] and we split them.
    """

    def __init__(self, num_model_params: int, server_lr: float = 1.0):
        self.server_lr = server_lr
        self.num_model_params = num_model_params
        self.global_control: Optional[List[np.ndarray]] = None

    def _split_params_and_delta_c(
        self, ndarrays: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split concatenated [model_params, delta_c] arrays."""
        model_params = ndarrays[: self.num_model_params]
        delta_c = ndarrays[self.num_model_params :]
        return model_params, delta_c

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        num_clients = len(results)

        all_model_params = []
        all_delta_c = []

        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            model_params, delta_c = self._split_params_and_delta_c(client_ndarrays)
            all_model_params.append((model_params, fit_res.num_examples))
            if delta_c:
                all_delta_c.append(delta_c)

        # Weighted average of model parameters
        first_params = all_model_params[0][0]
        aggregated = [np.zeros_like(p) for p in first_params]
        for params, n_examples in all_model_params:
            weight = n_examples / total_examples
            for i, p in enumerate(params):
                aggregated[i] += p * weight

        # Update global control variate: c = c + (1/N) * sum(delta_c_k)
        if all_delta_c:
            if self.global_control is None:
                self.global_control = [np.zeros_like(p) for p in all_delta_c[0]]

            for delta_c in all_delta_c:
                for i, dc in enumerate(delta_c):
                    self.global_control[i] += dc / num_clients

        parameters = ndarrays_to_parameters(aggregated)

        metrics: Dict[str, Scalar] = {
            "num_clients": num_clients,
            "num_failures": len(failures),
        }
        return parameters, metrics

    def get_global_control(self) -> Optional[List[np.ndarray]]:
        """Return the current global control variate for sending to clients."""
        return self.global_control

    def get_name(self) -> str:
        return "scaffold"
