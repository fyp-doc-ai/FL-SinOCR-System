"""FedAvg -- Federated Averaging aggregation strategy.

McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.
"""

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from fl_server.aggregators.base import AggregatorInterface, weighted_average


class FedAvgAggregator(AggregatorInterface):
    """Standard Federated Averaging: weighted mean of client parameters."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        aggregated_ndarrays = weighted_average(results)
        parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics: Dict[str, Scalar] = {
            "num_clients": len(results),
            "num_failures": len(failures),
        }
        return parameters, metrics

    def get_name(self) -> str:
        return "fedavg"
