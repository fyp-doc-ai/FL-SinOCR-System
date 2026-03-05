import flwr as fl
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import requests
import json
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

base_url = 'http://127.0.0.1:8000/'

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            aggregated_ndlists = []
            for item in aggregated_ndarrays:
                aggregated_ndlists.append(item.tolist())
            # Update weights in the backend app
            response = requests.post(base_url+'set-params', json={'params': json.dumps(aggregated_ndlists)})
        return aggregated_parameters, aggregated_metrics

# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)