import torch

from accdfl.core.models import Model


class LinearModel(Model):

    def __init__(self, input_dim: int, output_dim: int):
        super(LinearModel, self).__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
