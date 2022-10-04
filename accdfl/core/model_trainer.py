import logging
import os
import time
from asyncio import sleep

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from accdfl.core.datasets import create_dataset, Dataset
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import SessionSettings


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, settings: SessionSettings, participant_index):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings

        if settings.dataset in ["cifar10", "cifar10_niid", "mnist", "movielens"]:
            train_dir = data_dir
        else:
            train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset: Dataset = create_dataset(settings, participant_index=participant_index, train_dir=train_dir)

    async def train(self, model) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.debug("Device for training: %s", device)
        model.to(device)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum)
        train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // self.settings.learning.batch_size
        if len(train_set.dataset) % self.settings.learning.batch_size != 0:
            local_steps += 1

        self.logger.info("Will perform %d local steps of training (batch size: %d)",
                         local_steps, self.settings.learning.batch_size)

        start_time = time.time()
        for local_step in range(local_steps):
            try:
                data, target = next(train_set_it)
                model.train()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                optimizer.optimizer.zero_grad()
                self.logger.info('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
                output = model.forward(data)

                if self.settings.dataset == "movielens":
                    lossf = MSELoss()
                elif self.settings.dataset in ["cifar10", "cifar10_niid"]:
                    lossf = NLLLoss()
                else:
                    lossf = CrossEntropyLoss()

                loss = lossf(output, target)
                self.logger.info('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
                loss.backward()
                optimizer.optimizer.step()
            except StopIteration:
                pass

        if self.settings.is_simulation:
            # If we're running a simulation, we should advance the time of the DiscreteLoop with the elapsed real-world
            # time for training.
            elapsed_time = time.time() - start_time
            await sleep(elapsed_time)

        return model
