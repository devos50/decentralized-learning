import hashlib
import io
from typing import Optional, Tuple

import torch
from torch import Tensor


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
    Class to store/retrieve models in Tensor form.
    """

    def __init__(self):
        self.models = {}

    def add(self, model_state_dict):
        b = io.BytesIO()
        torch.save(model_state_dict, b)
        b.seek(0)
        serialized_model = b.read()

        h = hashlib.md5(serialized_model).digest()
        self.models[h] = serialized_model
