import os

import pytest

from accdfl.core.dataset import TrainDataset
from accdfl.core.model import create_model
from accdfl.core.model_manager import ModelManager
from accdfl.core.optimizer.sgd import SGDOptimizer


@pytest.fixture
def model_manager():
    parameters = {
        "batch_size": 25,
        "target_participants": 1,
        "dataset": "cifar10",
        "nodes_per_class": [1] * 10,
        "samples_per_class": [50] * 10,
        "local_classes": 10,
        "participants": ["a"]
    }
    model = create_model(parameters["dataset"], "gnlenet")
    dataset = TrainDataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters, 0)
    optimizer = SGDOptimizer(model, 0.002, 0.9)
    return ModelManager(model, dataset, optimizer, parameters)


def test_train(model_manager):
    assert model_manager.train() == 20
