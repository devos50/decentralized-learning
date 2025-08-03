from typing import Dict, List

from peft import PeftModel, set_peft_model_state_dict
import torch

from accdfl.core.gradient_aggregation import GradientAggregation


class FedAvg(GradientAggregation):

    @staticmethod
    def aggregate(adapters: List[Dict], peft_model: PeftModel) -> None:
        agg_state = {}
        for k in adapters[0].keys():
            stack = torch.stack([sd[k] for sd in adapters], dim=0)
            avg = stack.mean(dim=0)
            agg_state[k] = avg

        set_peft_model_state_dict(peft_model, agg_state, adapter_name="global")
