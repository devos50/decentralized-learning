from abc import abstractmethod
from enum import IntEnum
from typing import Dict, List

from peft import PeftModel
from torch import nn


def get_aggregator(name: str, peft_model: PeftModel, global_adapter: Dict) -> 'GradientAggregation':
    """
    Get the aggregator class by name.
    """
    if name == "fedadam":
        from accdfl.core.gradient_aggregation.fedadam import FedAdam
        return FedAdam(peft_model, global_adapter)
    elif name == "fedavg":
        from accdfl.core.gradient_aggregation.fedavg import FedAvg
        return FedAvg(peft_model, global_adapter)
    elif name == "fednesterov":
        from accdfl.core.gradient_aggregation.fednesterov import FedNesterov
        return FedNesterov(peft_model, global_adapter)
    else:
        raise ValueError(f"Unknown aggregator: {name}")


class GradientAggregation:

    @abstractmethod
    def aggregate(self, models: List[nn.Module], weights: List[float]):
        pass
