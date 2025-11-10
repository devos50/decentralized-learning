from transformers import PreTrainedModel
from accdfl.core.gradient_aggregation.fedopt import FedOpt

import torch

from accdfl.core.session_settings import SessionSettings


class FedNesterov(FedOpt):
    """
    Server-side Nesterov Accelerated Gradient.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        settings: SessionSettings,
    ):
        self.opt = torch.optim.SGD(
            model.parameters(), lr=settings.learning.server_learning_rate, momentum=settings.learning.server_momentum, nesterov=True
        )
        self.model = model
        self.settings = settings
