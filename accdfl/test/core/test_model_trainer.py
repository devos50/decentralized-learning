import os

import pytest

from accdfl.core.model import create_model
from accdfl.core.model_trainer import ModelTrainer


@pytest.fixture
def parameters():
    return {
        "batch_size": 20,
        "target_participants": 100,
        "dataset": "cifar10",
        "nodes_per_class": [1] * 10,
        "samples_per_class": [50] * 10,
        "local_classes": 10,
        "participants": ["a"],
        "learning_rate": 0.002,
        "momentum": 0.9,
    }


@pytest.fixture
def model(parameters):
    return create_model(parameters["dataset"])


@pytest.fixture
def model_trainer(parameters):
    return ModelTrainer(os.path.join(os.environ["HOME"], "dfl-data"), parameters, 0)


def test_train(parameters, model, model_trainer):
    model_trainer.train(model)
