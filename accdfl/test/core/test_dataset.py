from accdfl.core.datasets.partition import split_dataset_uniform
import pytest

from datasets import DatasetDict

from accdfl.core.datasets import create_global_dataset
from accdfl.core.session_settings import LearningSettings, SessionSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="ag_news",
        learning=LearningSettings(batch_size=20, learning_rate=0.002, momentum=0.9, weight_decay=0, local_steps=5),
        participants=["a"] * 10,  # Simulating 10 participants
        all_participants=["a"] * 10,
        target_participants=10
    )

def test_create_and_split_dataset(settings):
    dataset = create_global_dataset(settings)
    assert dataset is not None
    assert isinstance(dataset, DatasetDict)
    assert "train" in dataset
    assert "test" in dataset
    assert len(dataset["train"]) > 0
    assert len(dataset["test"]) > 0
    assert "text" in dataset["train"].features
    assert "label" in dataset["train"].features
    assert dataset["train"].features["label"].num_classes == 4

    # Partition the dataset
    split_datasets = split_dataset_uniform(dataset["train"], len(settings.participants))
    assert len(split_datasets) == len(settings.participants)
