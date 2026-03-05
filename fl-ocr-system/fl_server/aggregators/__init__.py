"""FL aggregation strategies: FedAvg, SCAFFOLD, FedOPT."""

from fl_server.aggregators.base import AggregatorInterface
from fl_server.aggregators.fedavg import FedAvgAggregator
from fl_server.aggregators.fedopt import FedOptAggregator
from fl_server.aggregators.scaffold import ScaffoldAggregator

AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "scaffold": ScaffoldAggregator,
    "fedopt": FedOptAggregator,
}

__all__ = [
    "AggregatorInterface",
    "FedAvgAggregator",
    "ScaffoldAggregator",
    "FedOptAggregator",
    "AGGREGATOR_REGISTRY",
]
