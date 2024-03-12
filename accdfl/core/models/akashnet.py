from torch import nn
import torch.nn.functional as F

from accdfl.core.models import Model


class AkashNet(Model):

    def __init__(self, d=4, total_clients=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(total_clients * num_classes, total_clients * d)
        self.fc2 = nn.Linear(total_clients * d, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
