from typing import Dict
from accdfl.core.gradient_aggregation.fedopt import FedOpt
from peft import PeftModel

import torch


class FedNesterov(FedOpt):
    """
    Server-side Nesterov Accelerated Gradient that updates *only* the global LoRA adapter inside a
    PEFT model which also contains the per-client adapters.  Single process safe.
    """
    def __init__(
        self,
        peft_model: PeftModel,
        global_adapter: Dict,
        lr: float = 5e-3,
    ):
        self.global_name = global_adapter["name"]

        # ==== pick *only* parameters belonging to the global adapter =========
        global_params = [
            p for n, p in peft_model.named_parameters()
            if p.requires_grad and self.global_name in n
        ]

        self.opt = torch.optim.SGD(
            global_params, lr=lr, momentum=0.9, nesterov=True
        )
        self.peft_model = peft_model
        self.gkeys = set(global_adapter["keys"])
