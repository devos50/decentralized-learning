import pickle

import torch

from accdfl.core.model.linear import LinearModel
from accdfl.core.model.gn_lenet import GNLeNet


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def create_model(dataset: str, model_type: str):
    if dataset == "mnist" and model_type == "linear":
        return LinearModel(28 * 28)
    elif dataset == "mnist" and model_type == "gnlenet":
        return GNLeNet(input_channel=1, output=10, model_input=(24, 24))


def unserialize_model(serialized_model: bytes, dataset: str, model_type: str) -> torch.nn.Module:
    model = create_model(dataset, model_type)
    model.load_state_dict(pickle.loads(serialized_model))
    return model
