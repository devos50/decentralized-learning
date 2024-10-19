import pickle
from typing import Optional

import torch

from accdfl.core.models.Model import Model


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def serialize_chunk(chunk) -> bytes:
    return chunk.numpy().tobytes()


def unserialize_model(serialized_model: bytes, dataset: str, architecture: Optional[str] = None) -> torch.nn.Module:
    model = create_model(dataset, architecture=architecture)
    model.load_state_dict(pickle.loads(serialized_model))
    return model


def create_model(dataset: str, architecture: Optional[str] = None) -> Model:
    if dataset in ["shakespeare", "shakespeare_sub", "shakespeare_sub96"]:
        from accdfl.core.models.shakespeare import LSTM
        return LSTM()
    elif dataset == "cifar10":
        if not architecture:
            from accdfl.core.models.cifar10 import GNLeNet
            return GNLeNet(input_channel=3, output=10, model_input=(32, 32))
        elif architecture == "resnet8":
            from accdfl.core.models.resnet8 import ResNet8
            return ResNet8()
        elif architecture in ["resnet18", "mobilenet_v3_large"]:
            import torchvision.models as tormodels
            return tormodels.__dict__[architecture](num_classes=10)
        else:
            raise RuntimeError("Unknown model architecture for CIFAR10: %s" % architecture)
    elif dataset == "celeba":
        from accdfl.core.models.celeba import CNN
        return CNN()
    elif dataset == "femnist":
        from accdfl.core.models.femnist import CNN
        return CNN()
    elif dataset == "movielens":
        from accdfl.core.models.movielens import MatrixFactorization
        return MatrixFactorization()
    elif dataset == "spambase":
        from accdfl.core.models.linear import LinearModel
        return LinearModel(57, 2)
    elif dataset == "google_speech":
        from accdfl.core.models.resnet_speech import resnet34
        return resnet34(num_classes=35, in_channels=1)
    else:
        raise RuntimeError("Unknown dataset %s" % dataset)
