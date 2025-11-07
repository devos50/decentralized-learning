from typing import Dict, Optional

from accdfl.diloco.reduction_manager import ReductionManager


class Round:

    def __init__(self, round_nr: int):
        self.round_nr: int = round_nr
        self.reduction_manager: Optional[ReductionManager] = None

        # It could be that we receive chunks out of order, for example, before the round starts.
        # In that situation, store the chunk to process it later.
        self.out_of_order_chunks: Dict = {}
