import torch


class SGDOptimizer:

    def __init__(self, model, learning_rate, momentum):
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum)
