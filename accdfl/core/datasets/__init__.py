import os
from typing import Optional, Dict

from accdfl.core.datasets.Dataset import Dataset
from accdfl.core.mappings import Linear


def create_dataset(parameters: Dict, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, parameters["target_participants"])
    if parameters["dataset"] in ["shakespeare", "shakespeare_sub", "shakespeare_sub96"]:
        from accdfl.core.datasets.Shakespeare import Shakespeare
        return Shakespeare(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "cifar10":
        from accdfl.core.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "celeba":
        from accdfl.core.datasets.Celeba import Celeba
        img_dir = None
        if train_dir:
            img_dir = os.path.join(train_dir, "..", "..", "data", "raw", "img_align_celeba")
        elif test_dir:
            img_dir = os.path.join(test_dir, "..", "raw", "img_align_celeba")
        return Celeba(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir, images_dir=img_dir)
    elif parameters["dataset"] == "femnist":
        from accdfl.core.datasets.Femnist import Femnist
        return Femnist(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif parameters["dataset"] == "movielens":
        from accdfl.core.datasets.MovieLens import MovieLens
        data_dir = train_dir or test_dir
        return MovieLens(participant_index, 0, mapping, train_dir=data_dir, test_dir=data_dir)
    else:
        raise RuntimeError("Unknown dataset %s" % parameters["dataset"])