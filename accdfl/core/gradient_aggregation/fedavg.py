from typing import Dict, List

import torch

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    @staticmethod
    def aggregate(adapters: List[Dict], global_adapter: Dict) -> None:
        for k in adapters[0].keys():
            stack = torch.stack([sd[k] for sd in adapters], dim=0)
            avg = stack.mean(dim=0)
            global_adapter[k].copy_(avg)
