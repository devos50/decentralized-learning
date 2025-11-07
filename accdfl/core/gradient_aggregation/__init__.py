from abc import abstractmethod
from typing import List

from torch import nn
from transformers import PreTrainedModel


def get_server_optimizer(name: str, model: PreTrainedModel) -> 'GradientAggregation':
    """
    Get the aggregator class by name.
    """
    if name == "nesterov":
        from accdfl.core.gradient_aggregation.fednesterov import FedNesterov
        return FedNesterov(model)
    else:
        raise ValueError(f"Unknown aggregator: {name}")


class GradientAggregation:

    @abstractmethod
    def aggregate(self, models: List[nn.Module], weights: List[float]):
        pass
