from abc import abstractmethod
from typing import List

from torch import nn


class GradientAggregation:

    @staticmethod
    @abstractmethod
    def aggregate(models: List[nn.Module]):
        pass
