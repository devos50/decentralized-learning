import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accdfl.core.datasets.Data import Data
from accdfl.core.session_settings import LearningSettings

logger = logging.getLogger("distillation")


def loss_fn_kd(outputs, teacher_outputs, settings: LearningSettings):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2

    Taken from https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py
    """
    T = settings.kd_temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * T * T

    return KD_loss


def get_random_images_data_loader(num_images: int, batch_size: int):
    rand = random.Random(42)
    x_data = []
    for img_ind in range(num_images):
        if img_ind % 1000 == 0:
            logger.debug("Generated %d images..." % img_ind)

        img = []
        for channel in range(3):
            channel_data = []
            for y in range(32):
                row = []
                for x in range(32):
                    row.append(rand.random())
                channel_data.append(row)
            img.append(channel_data)
        x_data.append(img)

    return DataLoader(Data(x=torch.tensor(x_data), y=torch.tensor([1] * num_images)), batch_size=batch_size, shuffle=True)
