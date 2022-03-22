import pickle

import torch

from accdfl.core.model.linear import LinearModel


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def unserialize_model(serialized_model: bytes) -> torch.nn.Module:
    model = LinearModel(28 * 28)
    model.load_state_dict(pickle.loads(serialized_model))
    return model
