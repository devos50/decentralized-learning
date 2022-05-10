from typing import Optional, Dict

from accdfl.core.datasets.Dataset import Dataset
from accdfl.core.mappings import Linear


def create_dataset(parameters: Dict, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, parameters["target_participants"])
    if parameters["dataset"] == "shakespeare":
        from accdfl.core.datasets.Shakespeare import Shakespeare
        return Shakespeare(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "cifar10":
        from accdfl.core.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "celeba":
        from accdfl.core.datasets.Celeba import Celeba
        return Celeba(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "femnist":
        from accdfl.core.datasets.Femnist import Femnist
        return Femnist(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "movielens":
        from accdfl.core.datasets.MovieLens import MovieLens
        return MovieLens(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    else:
        raise RuntimeError("Unknown dataset %s" % parameters["dataset"])
