import pickle

import torch

from accdfl.core.datasets.Shakespeare import LSTM
from accdfl.core.model.linear import LinearModel
from accdfl.core.model.gn_lenet import GNLeNet


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def create_model(dataset: str):
    if dataset == "shakespeare":
        return LSTM()
    else:
        raise RuntimeError("Unknown dataset %s" %dataset)


def unserialize_model(serialized_model: bytes, dataset: str) -> torch.nn.Module:
    model = create_model(dataset)
    model.load_state_dict(pickle.loads(serialized_model))
    return model
