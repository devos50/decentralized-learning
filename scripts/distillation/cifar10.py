"""
Test to distill knowledge from a CIFAR10 pre-trained model to another one.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import LearningSettings, SessionSettings


def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2

    Taken from https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py
    """
    alpha = 0.95
    T = 6
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

# Define settings
learning_settings = LearningSettings(
    learning_rate=0.001,
    momentum=0.9,
    batch_size=200
)

settings = SessionSettings(
    dataset="cifar10",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=1,
)

# Load a pre-trained CIFAR10 model
teacher_model = create_model("cifar10")
teacher_model.load_state_dict(torch.load("../cifar10.model"))

# Test accuracy
data_dir = os.path.join(os.environ["HOME"], "dfl-data")

mapping = Linear(1, 1)
s = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print(s.test(teacher_model))

# Create the student model
student_model = create_model("cifar10")
trainer = ModelTrainer(data_dir, settings, 0)

device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print("Device to train on: %s" % device)

# Determine outputs of the teacher model on the public training data
for epoch in range(4):
    optimizer = SGDOptimizer(student_model, settings.learning.learning_rate, settings.learning.momentum)
    train_set = trainer.dataset.get_trainset(batch_size=settings.learning.batch_size, shuffle=True)
    train_set_it = iter(train_set)
    local_steps = len(train_set.dataset) // settings.learning.batch_size
    if len(train_set.dataset) % settings.learning.batch_size != 0:
        local_steps += 1
    print("Will perform %d local steps" % local_steps)
    for local_step in range(local_steps):
        data, target = next(train_set_it)
        data, target = Variable(data.to(device)), Variable(target.to(device))

        print('d-sgd.next node forward propagation (step %d/%d)' % (local_step, local_steps))
        teacher_output = teacher_model.forward(data)
        student_output = student_model.forward(data)
        loss = loss_fn_kd(student_output, target, teacher_output)

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()

    acc, loss = s.test(student_model)
    print("Accuracy after %d epochs: %f, %f" % (epoch + 1, acc, loss))
