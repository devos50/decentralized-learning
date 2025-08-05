from typing import Dict, List

import torch

from peft import PeftModel

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    def aggregate(self, adapters: List[Dict], global_adapter: Dict, peft_model: PeftModel) -> None:
        state_dict = peft_model.state_dict()
        for k in global_adapter["keys"]:
            stack = torch.stack([state_dict[k.replace(global_adapter["name"], adapter["name"])] for adapter in adapters], dim=0)
            avg = stack.mean(dim=0)
            state_dict[k].copy_(avg)
