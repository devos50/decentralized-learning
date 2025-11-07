from transformers import PreTrainedModel
from accdfl.core.gradient_aggregation.fedopt import FedOpt

import torch


class FedNesterov(FedOpt):
    """
    Server-side Nesterov Accelerated Gradient.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        lr: float = 5e-3,
    ):
        self.opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True
        )
        self.model = model
