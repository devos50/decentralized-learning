from abc import abstractmethod
from enum import IntEnum
from typing import List

from torch import nn


def get_aggregator(name: str) -> 'GradientAggregation':
    """
    Get the aggregator class by name.
    """
    if name == "fedadam":
        from accdfl.core.gradient_aggregation.fedadam import FedAdam
        return FedAdam()
    elif name == "fedavg":
        from accdfl.core.gradient_aggregation.fedavg import FedAvg
        return FedAvg()
    else:
        raise ValueError(f"Unknown aggregator: {name}")


class GradientAggregation:

    @abstractmethod
    def aggregate(self, models: List[nn.Module], weights: List[float]):
        pass
