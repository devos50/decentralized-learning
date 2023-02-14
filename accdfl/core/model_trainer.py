import logging
import os
import random
import time
from asyncio import sleep
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from accdfl.core.datasets import create_dataset, Dataset
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import SessionSettings


def loss_fn_kd(outputs, teacher_outputs, T: float):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    Taken from https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py
    """
    return nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * T * T


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

    async def train(self, model, device_name: str = "cpu", proxy_dataset=None, predictions: Optional[Tuple[List[Tensor], List[int]]] = None) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        device = torch.device(device_name)
        model.to(device)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum)
        train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // self.settings.learning.batch_size
        if len(train_set.dataset) % self.settings.learning.batch_size != 0:
            local_steps += 1

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, data points: %d)",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, len(train_set.dataset))

        start_time = time.time()
        samples_trained_on = 0
        total_local_loss = 0
        total_distill_loss = 0
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
                total_local_loss += loss

                # Create the proxy data based on the indices
                proxy_data = []
                for sample_ind in predictions[1][samples_trained_on:samples_trained_on+self.settings.learning.batch_size]:
                    proxy_data.append(proxy_dataset.dataset[sample_ind][0])

                proxy_data = torch.stack(proxy_data)
                proxy_data = Variable(proxy_data.to(device))
                output = model.forward(proxy_data)
                sub_predictions = torch.stack(predictions[0][samples_trained_on:samples_trained_on+self.settings.learning.batch_size])
                dist_loss = loss_fn_kd(output, sub_predictions, 3) * 200
                total_distill_loss += dist_loss
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

        total_local_loss /= float(samples_trained_on)
        total_distill_loss /= float(samples_trained_on)
        return samples_trained_on, total_local_loss, total_distill_loss
