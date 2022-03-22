import pickle

import torch

from accdfl.core.model.linear import LinearModel


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model)


def unserialize_model(serialized_model: bytes) -> torch.nn.Module:
    return pickle.loads(serialized_model)
