"""
Test to distill knowledge from a CIFAR10 pre-trained model to another one.
"""
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.datasets.Data import Data
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import LearningSettings, SessionSettings


NUM_ROUNDS = 100 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cifar10_distillation")


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


if __name__ == "__main__":
    learning_settings = LearningSettings(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=200,
        kd_temperature=6 if "TEMPERATURE" not in os.environ else int(os.environ["TEMPERATURE"]),
    )

    settings = SessionSettings(
        dataset="cifar100",
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    with open(os.path.join("data", "accuracies.csv"), "w") as out_file:
        out_file.write("round,temperature,accuracy,loss\n")

    # Load a pre-trained CIFAR10 model
    teacher_model = create_model("cifar10")
    teacher_model_path = "../cifar10.model" if "TEACHER_MODEL" not in os.environ else os.environ["TEACHER_MODEL"]
    teacher_model.load_state_dict(torch.load(teacher_model_path))

    # Test accuracy
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")

    mapping = Linear(1, 1)
    cifar10_testset = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

    # Create the student model
    student_model = create_model("cifar10")
    #trainer = ModelTrainer(data_dir, settings, 0)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    logger.debug("Device to train on: %s", device)
    teacher_model.to(device)
    student_model.to(device)

    acc, loss = cifar10_testset.test(teacher_model, device_name=device)
    print("Teacher model accuracy: %f, loss: %f" % (acc, loss))

    # train_set = trainer.dataset.get_trainset(batch_size=settings.learning.batch_size, shuffle=True)
    train_set = get_random_images_data_loader(50000, settings.learning.batch_size)

    # Determine outputs of the teacher model on the public training data
    for epoch in range(NUM_ROUNDS):
        optimizer = SGDOptimizer(student_model, settings.learning.learning_rate, settings.learning.momentum)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // settings.learning.batch_size
        if len(train_set.dataset) % settings.learning.batch_size != 0:
            local_steps += 1
        logger.info("Will perform %d local steps", local_steps)
        for local_step in range(local_steps):
            data, _ = next(train_set_it)
            data = Variable(data.to(device))

            logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
            teacher_output = teacher_model.forward(data)
            student_output = student_model.forward(data)
            loss = loss_fn_kd(student_output, teacher_output, learning_settings)

            optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.optimizer.step()

        acc, loss = cifar10_testset.test(student_model, device_name=device)
        print("Accuracy after %d epochs: %f, %f" % (epoch + 1, acc, loss))
        with open(os.path.join("data", "accuracies.csv"), "a") as out_file:
            out_file.write("%d,%d,%f,%f\n" % (epoch + 1, learning_settings.kd_temperature, acc, loss))
