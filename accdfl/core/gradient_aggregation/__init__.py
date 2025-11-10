from abc import abstractmethod
from typing import List

from torch import nn
from transformers import PreTrainedModel

from accdfl.core.session_settings import SessionSettings


def get_server_optimizer(model: PreTrainedModel, settings: SessionSettings) -> 'GradientAggregation':
    """
    Get the aggregator class by name.
    """
    if settings.learning.server_optimizer == "nesterov":
        from accdfl.core.gradient_aggregation.fednesterov import FedNesterov
        return FedNesterov(model, settings)
    else:
        raise ValueError(f"Unknown aggregator: {settings.learning.server_optimizer}")


class GradientAggregation:

    @abstractmethod
    def aggregate(self, models: List[nn.Module], weights: List[float]):
        pass
