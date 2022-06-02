import torch
import torch.nn.functional as F


class LinearModel(torch.nn.Module):
    """ Network architecture. """

    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.fc = torch.nn.Linear(input_size,10)
        self.input_size = input_size

    def forward(self, x):
        x = self.fc(x.view(-1, self.input_size))
        return F.log_softmax(x, dim=1)

    def copy(self):
        c = LinearModel(self.input_size)
        for c1, s1 in zip(c.parameters(), self.parameters()):
            c1.mul_(0)
            c1.add_(s1)
        return c
