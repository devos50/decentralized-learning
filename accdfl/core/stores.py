import hashlib
import io
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


class DataType(Enum):
    TRAIN_DATA = 0
    MODEL = 1


class DataStore:
    """
    Class to store/retrieve training data in Tensor form.
    """

    def __init__(self):
        self.data_items = {}

    def add(self, data: Tensor, target: Tensor) -> None:
        h = hashlib.md5(b"%d" % hash(data)).digest()
        self.data_items[h] = (data, target)

    def get(self, data_hash: bytes) -> Optional[Tuple[Tensor]]:
        return self.data_items.get(data_hash, None)


class ModelStore:
    """
    Class to store/retrieve models in serialized form.
    """

    def __init__(self):
        self.models = {}

    def add(self, serialized_model: bytes) -> None:
        h = hashlib.md5(serialized_model).digest()
        self.models[h] = serialized_model

    def get(self, model_hash: bytes) -> Optional[bytes]:
        return self.models.get(model_hash, None)
