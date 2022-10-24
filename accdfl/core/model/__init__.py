import pickle

import torch


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def create_model(dataset: str):
    if dataset in ["shakespeare", "shakespeare_sub", "shakespeare_sub96"]:
        from accdfl.core.datasets.Shakespeare import LSTM
        return LSTM()
    elif dataset == "cifar10" or dataset == "cifar10_niid":
        from accdfl.core.models.cifar10 import GNLeNet
        return GNLeNet(input_channel=3, output=10, model_input=(32, 32))
    elif dataset == "celeba":
        from accdfl.core.models.celeba import CNN
        return CNN()
    elif dataset == "femnist":
        from accdfl.core.datasets.Femnist import CNN
        return CNN()
    elif dataset == "movielens":
        from accdfl.core.datasets.MovieLens import MatrixFactorization
        return MatrixFactorization()
    else:
        raise RuntimeError("Unknown dataset %s" %dataset)


def unserialize_model(serialized_model: bytes, dataset: str) -> torch.nn.Module:
    model = create_model(dataset)
    model.load_state_dict(pickle.loads(serialized_model))
    return model
