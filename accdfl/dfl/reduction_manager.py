from asyncio import Future
from typing import List, Optional

import torch


class ReductionManager:

    def __init__(self, round: int, model, participants_ids: List[str], my_rank: int):
        self.round: int = round
        self.model = model
        self.participants_ids: List[str] = participants_ids
        self.my_rank: int = my_rank
        self.chunks: List = [None] * len(self.participants_ids)
        self.step: int = 0
        self.receive_future: Optional[Future] = None

    def prepare(self):
        # Chunk
        flat_params = ReductionManager.get_flat_params(self.model)
        total_elements = flat_params.numel()
        chunk_size = total_elements // len(self.participants_ids)
        self.chunks = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(len(self.participants_ids))]

        # Handle any remaining elements
        if total_elements % len(self.participants_ids) != 0:
            remaining = flat_params[len(self.participants_ids) * chunk_size:]
            self.chunks[-1] = torch.cat([self.chunks[-1], remaining])

    def finish(self):
        # Reconstruct the flat tensor
        flat_params = ReductionManager.get_flat_params(self.model)
        aggregated_flat_params = torch.empty_like(flat_params)
        for i, chunk in enumerate(self.chunks):
            aggregated_flat_params[i::len(self.participants_ids)] = chunk
        # TODO
        a = 1 / 0

    def process_received_chunk(self, chunk_idx: int, chunk: List):
        self.chunks[chunk_idx] += chunk
        self.step += 1
        self.receive_future.set_result(None)

    @staticmethod
    def get_flat_params(model):
        param_tensors = [param.data.view(-1) for param in model.parameters()]
        flat_params = torch.cat(param_tensors)
        return flat_params
    
    def get_chunk_to_send(self):
        idx: int = (self.my_rank - self.step) % len(self.participants_ids)
        return idx, self.chunks[idx]
