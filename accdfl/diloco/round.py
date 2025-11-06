from typing import Optional

from accdfl.diloco.reduction_manager import ReductionManager


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.reduction_manager: Optional[ReductionManager] = None
