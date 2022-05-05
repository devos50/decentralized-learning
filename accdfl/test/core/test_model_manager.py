import pytest

from accdfl.core.model import create_model
from accdfl.core.model_manager import ModelManager


@pytest.fixture
def model_manager():
    parameters = {
        "batch_size": 25,
        "target_participants": 1,
        "dataset": "cifar10",
        "nodes_per_class": [1] * 10,
        "samples_per_class": [50] * 10,
        "local_classes": 10,
        "participants": ["a"],
        "learning_rate": 0.002,
        "momentum": 0.9,
        "data_distribution": "iid",
    }
    model = create_model(parameters["dataset"], "gnlenet")
    return ModelManager(model, parameters, 0)


async def test_train(model_manager):
    await model_manager.train()
