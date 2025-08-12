from typing import Dict, List

import torch

from peft import PeftModel

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    def __init__(self, peft_model: PeftModel, global_adapter: Dict):
        self.peft_model = peft_model
        self.global_adapter = global_adapter

    def aggregate(self, adapters: List[Dict]) -> None:
        state_dict = self.peft_model.state_dict()
        for k in self.global_adapter["keys"]:
            stack = torch.stack([state_dict[k.replace(self.global_adapter["name"], adapter["name"])] for adapter in adapters], dim=0)
            avg = stack.mean(dim=0)
            state_dict[k].copy_(avg)
