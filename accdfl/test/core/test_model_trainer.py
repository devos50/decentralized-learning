import os

import pytest

from accdfl.core.model import create_model
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import SessionSettings, LearningSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="cifar10",
        learning=LearningSettings(batch_size=20, learning_rate=0.002, momentum=0.9),
        participants=["a"],
        all_participants=["a"],
        target_participants=100
    )


@pytest.fixture
def model(settings):
    return create_model(settings.dataset)


@pytest.fixture
def model_trainer(settings):
    return ModelTrainer(os.path.join(os.environ["HOME"], "dfl-data"), settings, 0)


def test_train(settings, model, model_trainer):
    model_trainer.train(model)
