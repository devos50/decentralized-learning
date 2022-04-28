import torch
import torch.nn.functional as F

from accdfl.core.dataset import TestDataset


evaluator = None


def setup_evaluator(data_dir, parameters):
    global evaluator
    evaluator = Evaluator(data_dir, parameters)


def evaluate_accuracy(model):
    global evaluator
    return evaluator.evaluate_accuracy(model)


class Evaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    Runs in a separate process.
    """

    def __init__(self, data_dir, parameters):
        self.test_dataset = TestDataset(data_dir, parameters)

    def evaluate_accuracy(self, model):
        correct = example_number = total_loss = num_batches = 0
        model.eval()

        with torch.no_grad():
            for data, target in self.test_dataset.iterator:
                output = model.forward(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1

        accuracy = float(correct) / float(example_number)
        loss = total_loss / float(example_number)
        self.test_dataset.reset_iterator()
        return accuracy, loss
