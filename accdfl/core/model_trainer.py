import logging
import os

import torch.nn.functional as F
from torch.autograd import Variable

from accdfl.core.datasets.Shakespeare import Shakespeare
from accdfl.core.mappings import Linear
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

        train_dir = os.path.join(data_dir, "per_user_data", "train")

        if parameters["dataset"] == "shakespeare":
            mapping = Linear(1, parameters["target_participants"])
            self.dataset = Shakespeare(participant_index, 0, mapping, train_dir=train_dir)
        else:
            raise RuntimeError("Unknown dataset %s" % parameters["dataset"])

    def train(self, model) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        optimizer = SGDOptimizer(model, self.parameters["learning_rate"], self.parameters["momentum"])

        train_set = self.dataset.get_trainset(batch_size=self.parameters["batch_size"], shuffle=True)
        train_set_it = iter(train_set)
        local_steps = len(train_set) // self.parameters["batch_size"]
        if len(train_set) % self.parameters["batch_size"] != 0:
            local_steps += 1

        for local_step in range(local_steps):
            data, target = next(train_set_it)
            model.train()
            data, target = Variable(data), Variable(target)
            optimizer.optimizer.zero_grad()
            self.logger.info('d-sgd.next node forward propagation')
            output = model.forward(data)
            loss = F.nll_loss(output, target)
            self.logger.info('d-sgd.next node backward propagation')
            loss.backward()
            optimizer.optimizer.step()

        return model
