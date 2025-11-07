import copy
from asyncio import Future
from typing import Dict, List

import torch


class ReductionManager:

    def __init__(self, round: int, gradients: list, participants_ids: List[str], my_rank: int):
        self.round: int = round
        self.gradients = gradients
        self.participants_ids: List[str] = participants_ids
        self.my_rank: int = my_rank
        self.chunks: List = [None] * len(self.participants_ids)
        self.step: int = 0
        self.receive_futures: Dict[int, Future] = {}

    def prepare(self):
        # Chunk
        flat_params = ReductionManager.get_flat_params(self.gradients).cpu()
        total_elements = flat_params.numel()
        chunk_size = total_elements // len(self.participants_ids)
        self.chunks = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(len(self.participants_ids))]

        # Handle any remaining elements
        if total_elements % len(self.participants_ids) != 0:
            remaining = flat_params[len(self.participants_ids) * chunk_size:]
            self.chunks[-1] = torch.cat([self.chunks[-1], remaining])

    def get_aggregated_gradients(self):
        # Reconstruct the flat tensor
        flat = torch.cat(self.chunks)
        flat.div_(len(self.chunks))

        # Unflatten back into per-parameter tensors matching `self.gradients`
        grads_out: list[torch.Tensor] = []
        pointer = 0
        for g in self.gradients:
            numel = g.numel()
            # slice keeps same storage; clone to make an independent tensor
            slice_view = flat[pointer:pointer + numel].view(g.shape)
            grads_out.append(slice_view.clone().to(g.device).type_as(g))
            pointer += numel

        return grads_out

    def process_received_chunk(self, step: int, chunk_idx: int, chunk):
        self.chunks[chunk_idx].add_(chunk)
        self.receive_futures[step].set_result(None)

    @staticmethod
    def get_flat_params(gradients: list):
        param_tensors = [grad.data.view(-1) for grad in gradients]
        flat_params = torch.cat(param_tensors)
        return flat_params
    
    def get_chunk_to_send(self, step: int):
        idx: int = (self.my_rank - step) % len(self.participants_ids)
        return idx, self.chunks[idx].clone()
