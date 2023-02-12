import logging
import os
import random
import time
from asyncio import sleep
from typing import Optional, List

import torch
from torch import Tensor
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

        if settings.dataset in ["cifar10", "mnist", "movielens", "spambase"]:
            train_dir = data_dir
        else:
            train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset: Dataset = create_dataset(settings, participant_index=participant_index, train_dir=train_dir)

    async def train(self, model, device_name: str = "cpu", proxy_dataset=None, predictions: Optional[List[List[Tensor]]] = None, my_id: Optional[int] = None) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        device = torch.device(device_name)
        model.to(device)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum)
        train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
        train_set_it = iter(train_set)
        proxy_set_it = iter(proxy_dataset)
        local_steps = len(train_set.dataset) // self.settings.learning.batch_size
        if len(train_set.dataset) % self.settings.learning.batch_size != 0:
            local_steps += 1

        # Choose which peer we're going to distill from
        possibilities = [i for i in range(self.settings.target_participants) if i != my_id]
        peer_to_distill_from = random.choice(possibilities)

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, data points: %d) - peer %d will distill from peer %d",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, len(train_set.dataset), my_id, peer_to_distill_from)

        start_time = time.time()
        samples_trained_on = 0
        for local_step in range(local_steps):
            try:
                # First train on the local data
                data, target = next(train_set_it)
                model.train()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                optimizer.optimizer.zero_grad()
                self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
                output = model.forward(data)

                if self.settings.dataset == "movielens":
                    lossf = MSELoss()
                elif self.settings.dataset == "cifar10":
                    if self.settings.model == "resnet8":
                        lossf = CrossEntropyLoss()
                    else:
                        lossf = CrossEntropyLoss()
                else:
                    lossf = CrossEntropyLoss()

                loss = lossf(output, target)

                # Then obtain the distillation loss
                proxy_data, _, indices = next(proxy_set_it)
                proxy_data = Variable(proxy_data.to(device))
                # print("-- data --")
                # print(data[0])
                # print(target[0])
                output = model.forward(proxy_data)
                sub_predictions = torch.stack([predictions[peer_to_distill_from][ind].clone() for ind in indices])
                # print("-- output -- ")
                # print(output[0])
                # print(sub_predictions[0])
                dist_loss = torch.nn.L1Loss()(output, sub_predictions)
                loss = loss + dist_loss

                self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
                loss.backward()
                optimizer.optimizer.step()
                samples_trained_on += len(data)
            except StopIteration:
                pass

        if self.settings.is_simulation:
            # If we're running a simulation, we should advance the time of the DiscreteLoop with the elapsed real-world
            # time for training. Otherwise,training would be instant.
            elapsed_time = time.time() - start_time
            await sleep(elapsed_time)

        return samples_trained_on
