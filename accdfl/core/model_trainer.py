import itertools
import logging

import torch.nn.functional as F
from torch.autograd import Variable

from accdfl.core.dataset import TrainDataset
from accdfl.core.optimizer.sgd import SGDOptimizer

trainer = None


def setup_trainer(data_dir, parameters, participant_index):
    global trainer
    trainer = ModelTrainer(data_dir, parameters, participant_index)


def train_model(model):
    global trainer
    return trainer.train(model)


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, parameters, participant_index):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parameters = parameters
        self.dataset = TrainDataset(data_dir, parameters, participant_index)

    def train(self, model) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        optimizer = SGDOptimizer(model, self.parameters["learning_rate"], self.parameters["momentum"])

        def it_has_next(iterable):
            try:
                first = next(iterable)
            except StopIteration:
                return None
            return itertools.chain([first], iterable)

        local_steps = len(self.dataset.train_set) // self.parameters["batch_size"]
        if len(self.dataset.train_set) % self.parameters["batch_size"] != 0:
            local_steps += 1

        for local_step in range(local_steps):
            data, target = self.dataset.iterator.__next__()
            model.train()
            data, target = Variable(data), Variable(target)
            optimizer.optimizer.zero_grad()
            self.logger.info('d-sgd.next node forward propagation')
            output = model.forward(data)
            loss = F.nll_loss(output, target)
            self.logger.info('d-sgd.next node backward propagation')
            loss.backward()
            optimizer.optimizer.step()

        # Are we at the end of the epoch?
        res = it_has_next(self.dataset.iterator)
        if res is None:
            self.logger.info("Epoch done - resetting dataset iterator")
            self.dataset.reset_train_iterator()
        else:
            self.dataset.iterator = res

        return model
