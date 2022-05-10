import os

import pytest

from accdfl.core.model import create_model
from accdfl.core.modelevaluator import ModelEvaluator


@pytest.fixture
def parameters():
    return {
        "batch_size": 20,
        "target_participants": 100,
        "dataset": "shakespeare",
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
def model_evaluator(parameters):
    return ModelEvaluator(os.path.join(os.environ["HOME"], "leaf", parameters["dataset"]), parameters)


def test_evaluate(parameters, model, model_evaluator):
    assert model_evaluator.evaluate_accuracy(model)
