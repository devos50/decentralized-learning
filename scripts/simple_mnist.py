import itertools
import logging
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms

from accdfl.core.dataset import Dataset
from accdfl.core.model import create_model
from accdfl.core.optimizer.sgd import SGDOptimizer

parameters = {
    "batch_size": 200,
    "participants": ['a'],
    "dataset": "mnist",
}

epoch = 0
logger = logging.getLogger()
logger.setLevel(logging.INFO)
model = create_model(parameters["dataset"])
optimizer = SGDOptimizer(model, learning_rate=0.1, momentum=0)
dataset = Dataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters, 0)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_test_dataset = datasets.MNIST(
    os.path.join(os.environ["HOME"], "dfl-data"),
    train=False,
    download=True,
    transform=mnist_transform)
mnist_test_iterator = iter(torch.utils.data.DataLoader(
    mnist_test_dataset,
    batch_size=parameters["batch_size"], shuffle=True))


def reset_test_iterator():
    global mnist_test_iterator
    mnist_test_iterator = iter(torch.utils.data.DataLoader(
        mnist_test_dataset,
        batch_size=parameters["batch_size"], shuffle=True))


def it_has_next(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


for step in range(50):
    logger.info("Step %d" % step)
    data, target = dataset.iterator.__next__()
    model.train()
    data, target = Variable(data), Variable(target)
    optimizer.optimizer.zero_grad()
    logger.info('d-sgd.next node forward propagation')
    output = model.forward(data)
    loss = F.nll_loss(output, target)
    logger.info('d-sgd.next node backward propagation')
    loss.backward()
    optimizer.optimizer.step()

    # Are we at the end of the epoch?
    res = it_has_next(dataset.iterator)
    if res is None:
        epoch += 1
        logger.info("Epoch done - resetting dataset iterator")
        dataset.reset_train_iterator()
    else:
        dataset.iterator = res

    if step % 25 == 0:
        logger.info("Testing accuracy...")
        # Test accuracy
        model.eval()
        correct = 0
        example_number = 0
        num_batches = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in mnist_test_iterator:
                output = model.forward(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1
        print(float(correct) / float(example_number))
        print(total_loss / float(example_number))

        reset_test_iterator()
