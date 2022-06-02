import pytest

from accdfl.core.model import create_model
from accdfl.core.model_manager import ModelManager


@pytest.fixture
def model_manager(tmpdir):
    parameters = {
        "batch_size": 20,
        "target_participants": 100,
        "dataset": "cifar10",
        "nodes_per_class": [1] * 10,
        "samples_per_class": [50] * 10,
        "local_classes": 10,
        "participants": ["a"],
        "learning_rate": 0.002,
        "momentum": 0.9,
        "work_dir": str(tmpdir),
    }
    model = create_model(parameters["dataset"])
    return ModelManager(model, parameters, 0)


@pytest.mark.timeout(10)
async def test_train(model_manager):
    await model_manager.train()


@pytest.mark.timeout(10)
async def test_compute_accuracy(model_manager):
    accuracy, loss = await model_manager.compute_accuracy(model_manager.model)
    assert accuracy > 0
    assert loss > 0
