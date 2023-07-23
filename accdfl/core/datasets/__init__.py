import os
from typing import Optional, Dict

from accdfl.core.datasets.Dataset import Dataset
from accdfl.core.mappings import Linear
from accdfl.core.session_settings import SessionSettings


def create_dataset(settings: SessionSettings, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, settings.target_participants)
    if settings.dataset in ["shakespeare", "shakespeare_sub", "shakespeare_sub96"]:
        from accdfl.core.datasets.Shakespeare import Shakespeare
        return Shakespeare(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir)
    elif settings.dataset == "cifar10":
        from accdfl.core.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, settings.partitioner,
                       train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants,
                       alpha=settings.alpha, validation_size=settings.validation_set_fraction, seed=settings.seed)
    elif settings.dataset == "cifar100":
        from accdfl.core.datasets.CIFAR100 import CIFAR100
        return CIFAR100(participant_index, 0, mapping, settings.partitioner,
                        train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants, alpha=settings.alpha)
    elif settings.dataset == "stl10":
        from accdfl.core.datasets.STL10 import STL10
        return STL10(participant_index, 0, mapping, settings.partitioner, train_dir=train_dir, test_dir=test_dir)
    elif settings.dataset == "celeba":
        from accdfl.core.datasets.Celeba import Celeba
        img_dir = None
        if train_dir:
            img_dir = os.path.join(train_dir, "..", "..", "data", "raw", "img_align_celeba")
        elif test_dir:
            img_dir = os.path.join(test_dir, "..", "raw", "img_align_celeba")
        return Celeba(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir, images_dir=img_dir)
    elif settings.dataset == "mnist":
        from accdfl.core.datasets.MNIST import MNIST
        return MNIST(participant_index, 0, mapping, settings.partitioner,
                     train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants, alpha=settings.alpha)
    elif settings.dataset == "fashionmnist":
        from accdfl.core.datasets.FashionMNIST import FashionMNIST
        return FashionMNIST(participant_index, 0, mapping, settings.partitioner,
                            train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants,
                            alpha=settings.alpha)
    elif settings.dataset == "femnist":
        from accdfl.core.datasets.Femnist import Femnist
        return Femnist(participant_index, 0, mapping, train_dir=train_dir, test_dir=test_dir,
                       validation_size=settings.validation_set_fraction)
    elif settings.dataset == "svhn":
        from accdfl.core.datasets.SVHN import SVHN
        return SVHN(participant_index, 0, mapping, settings.partitioner, train_dir=train_dir, test_dir=test_dir)
    elif settings.dataset == "movielens":
        from accdfl.core.datasets.MovieLens import MovieLens
        data_dir = train_dir or test_dir
        return MovieLens(participant_index, 0, mapping, train_dir=data_dir, test_dir=data_dir)
    elif settings.dataset == "spambase":
        from accdfl.core.datasets.spambase import Spambase
        data_dir = train_dir or test_dir
        return Spambase(participant_index, 0, mapping, settings.partitioner, train_dir=data_dir, test_dir=data_dir,
                        shards=settings.target_participants, alpha=settings.alpha)
    elif settings.dataset == "google_speech":
        from accdfl.core.datasets.google_speech import GoogleSpeech
        return GoogleSpeech(participant_index, 0, mapping, settings.partitioner,
                            train_dir=train_dir, test_dir=test_dir)
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
