"""Abstract base class for FL aggregation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class AggregatorInterface(ABC):
    """Interface that all FL aggregation strategies must implement."""

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client model updates after a training round."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the algorithm name."""
        ...


def weighted_average(
    results: List[Tuple[ClientProxy, FitRes]],
) -> List[np.ndarray]:
    """Compute weighted average of client parameters, weighted by num_examples."""
    total_examples = sum(fit_res.num_examples for _, fit_res in results)

    first_params = parameters_to_ndarrays(results[0][1].parameters)
    aggregated = [np.zeros_like(p) for p in first_params]

    for _, fit_res in results:
        weight = fit_res.num_examples / total_examples
        client_params = parameters_to_ndarrays(fit_res.parameters)
        for i, param in enumerate(client_params):
            aggregated[i] += param * weight

    return aggregated
