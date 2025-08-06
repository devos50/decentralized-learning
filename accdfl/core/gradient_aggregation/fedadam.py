from typing import Dict, List
from peft import PeftModel
from accdfl.core.gradient_aggregation import GradientAggregation

import torch


class FedAdam(GradientAggregation):
    """
    Server-side AdamW that updates *only* the global LoRA adapter inside a
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

        self.opt = torch.optim.AdamW(
            global_params, lr=lr,
        )
        self.peft_model = peft_model
        self.gkeys = set(global_adapter["keys"])

    # -------- aggregation ----------------------------------------------------
    @torch.no_grad()
    def aggregate(self, adapters: List[Dict]) -> None:
        """
        adapters – list like [{'name':'client_0','num_samples':1234}, …]
        """
        sd = self.peft_model.state_dict()

        # 1. weighted average delta
        total_w = sum(a.get("num_samples", 1.0) for a in adapters) + 1e-12
        avg_delta = {k: torch.zeros_like(sd[k]) for k in self.gkeys}

        for a in adapters:
            w = float(a.get("num_samples", 1.0)) / total_w
            for k in self.gkeys:
                ck = k.replace(self.global_name, a["name"])
                avg_delta[k] += (sd[k] - sd[ck]) * w  # global - client

        # 2. treat delta as gradient, AdamW step
        self.opt.zero_grad(set_to_none=False)
        for n, p in self.peft_model.named_parameters():
            if n in avg_delta:            # global adapter param
                p.grad = avg_delta[n]
        self.opt.step()
