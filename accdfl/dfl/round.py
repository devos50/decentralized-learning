from typing import Optional

from accdfl.dfl.reduction_manager import ReductionManager

from torch import nn


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.model: Optional[nn.Module] = None

        # It could be that we receive a chunk, even before the round starts.
        # In that situation, store the chunk to process it later.
        self.chunk_received_before_start = None

        self.reduction_manager: Optional[ReductionManager] = None

        # State
        self.is_training: bool = False
        self.train_done: bool = False
