import os

import pytest

from accdfl.core.dataset import TrainDataset


@pytest.fixture
def train_dataset_params():
    return {
        "batch_size": 25,
        "target_participants": 1,
        "dataset": "cifar10",
        "nodes_per_class": [1] * 10,
        "samples_per_class": [50] * 10,
        "local_classes": 10,
        "participants": ["a"],
        "learning_rate": 0.002,
        "momentum": 0.9,
        "data_distribution": "iid"
    }


@pytest.fixture
def train_dataset_iid(train_dataset_params):
    return TrainDataset(os.path.join(os.environ["HOME"], "dfl-data"), train_dataset_params, 0)


@pytest.fixture
def train_dataset_non_iid(train_dataset_params):
    train_dataset_params["local_shards"] = 5
    train_dataset_params["shard_size"] = 20
    train_dataset_params["participants"] = ["a", "b", "c", "d", "e"]
    train_dataset_params["data_distribution"] = "non-iid"
    return TrainDataset(os.path.join(os.environ["HOME"], "dfl-data"), train_dataset_params, 0)


def test_iid_train_dataset(train_dataset_iid):
    stats = train_dataset_iid.get_statistics()
    assert stats["total_samples"] == 500
    assert stats["samples_per_class"] == [50] * 10


def test_non_iid_train_dataset(train_dataset_non_iid):
    stats = train_dataset_non_iid.get_statistics()
    assert stats["total_samples"] == 100
    assert sum(stats["samples_per_class"]) == 100
