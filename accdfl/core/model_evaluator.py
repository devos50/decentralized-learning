import os

from accdfl.core.datasets import create_dataset
from accdfl.core.session_settings import SessionSettings


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    Runs in a separate process.
    """

    def __init__(self, data_dir: str, settings: SessionSettings):
        if settings.dataset in ["cifar10", "cifar10_niid", "mnist", "movielens"]:
            test_dir = data_dir
        else:
            test_dir = os.path.join(data_dir, "data", "test")
        self.dataset = create_dataset(settings, test_dir=test_dir)

    def evaluate_accuracy(self, model):
        model.eval()
        return self.dataset.test(model)
