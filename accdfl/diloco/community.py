from asyncio import Future, ensure_future
import asyncio
from binascii import hexlify, unhexlify
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accdfl.core.community import LearningCommunity
from accdfl.core.models import serialize_chunk
from accdfl.diloco.reduction_manager import ReductionManager
from accdfl.diloco.round import Round
from accdfl.util.eva.result import TransferResult
from pyipv8.ipv8.peer import Peer
from simulations.bandwidth_scheduler import BWScheduler


class DiLoCoCommunity(LearningCommunity):
    community_id = unhexlify('e5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round: int = 0
        self.round_info: Dict[int, Round] = {}
        self.incoming_adapters: List[Tuple[bytes, Dict]] = []  # Incoming adapters for a round
        self.nodes = None
        self.node_id: int = -1
        self.bandwidth: Optional[float] = None
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []

        self.bw_scheduler: BWScheduler = BWScheduler(self.my_peer.public_key.key_to_bin(),
                                                     self.peer_manager.get_my_short_id())

    def start(self):
        """
        Start to participate in the training process.
        """
        super().start()

    def start_round(self, round_nr: int):
        self.round = round_nr
        self.round_info[self.round] = Round(round_nr)
        self.register_task("round_%d" % round_nr, self.do_round)

    async def do_round(self):
        """
        Perform a single round. This method is expected to be called by a global coordinator.
        """
        self.logger.info("Peer %s starting round %d", self.peer_manager.get_my_short_id(), self.round)
        round_info: Round = self.round_info[self.round]

        # Train
        gradients, _ = await self.model_manager.train()

        # 2. Synchronize the gradients in a ring all-reduce fashion
        await self.do_ring_allreduce(self.round_info[self.round], gradients)

        aggregated_gradients = round_info.reduction_manager.get_aggregated_gradients()
        self.logger.info("Peer %s done with all-reduce in round %d", self.peer_manager.get_my_short_id(), round_info.round_nr)
        round_info.reduction_manager = None

        # Apply the outer optimizer
        self.model_manager.apply_outer_optimizer(aggregated_gradients)

        # Round completed!
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round_info.round_nr)
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(round_info.round_nr, self.model_manager.model))
        self.round_info.pop(round_info.round_nr)

    async def do_ring_allreduce(self, round_info: Round, gradients: list) -> None:
        round_nr: int = round_info.round_nr
        participants = [peer.public_key.key_to_bin() for peer in self.get_peers()]
        participants = sorted(participants)
        total_participants: int = len(participants)
        my_rank: int = participants.index(self.my_id)

        round_info.reduction_manager = ReductionManager(round, gradients, participants, my_rank)
        round_info.reduction_manager.prepare()

        # Prepare all futures
        for step in range(2 * total_participants - 1):
            round_info.reduction_manager.receive_futures[step] = Future()

        for step in range(2 * (total_participants - 1)):
            round_info.reduction_manager.step = step

            # Send chunk
            idx, chunk = round_info.reduction_manager.get_chunk_to_send(step)
            recipient_peer_pk = participants[(my_rank + 1) % len(participants)]
            peer = self.get_peer_by_pk(recipient_peer_pk)
            if not peer:
                raise RuntimeError("Could not find peer with public key %s", hexlify(recipient_peer_pk).decode())

            ensure_future(self.eva_send_chunk(round_nr, step, idx, chunk, peer))

            if step < total_participants - 1:
                round_info.reduction_manager.chunks[idx].zero_()

            # Check if we have this chunk
            if step in round_info.out_of_order_chunks:
                chunk_idx, rec_chunk = round_info.out_of_order_chunks[step]
                self.received_model_chunk(round_nr, step, chunk_idx, rec_chunk)
                round_info.out_of_order_chunks.pop(step)

            await round_info.reduction_manager.receive_futures[step]
    
    def eva_send_chunk(self, round: int, step: int, chunk_idx: int, chunk, peer):
        start_time = asyncio.get_event_loop().time()
        serialized_chunk = serialize_chunk(chunk)
        response = {"round": round, "step": step, "idx": chunk_idx, "type": "chunk"}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, serialized_chunk, start_time)
    
    def schedule_eva_send_model(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float) -> Future:
        future = ensure_future(self.bypass_send(peer, serialized_response, binary_data))
        future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, serialized_response, binary_data, start_time))
        return future
    
    async def bypass_send(self, peer: Peer, serialized_response: bytes, binary_data: bytes):
        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                transfer_start_time = asyncio.get_event_loop().time()
                transfer_size: int = len(binary_data) + len(serialized_response)
                if self.bw_scheduler.bw_limit > 0:
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size)
                    self.logger.info("Transfer %s => %s started at t=%f (size: %d)",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time, transfer_size)
                    try:
                        await transfer.complete_future
                    except RuntimeError:
                        transfer_success = False
                    transfer_time = asyncio.get_event_loop().time() - transfer_start_time

                    transferred_bytes: int = int(transfer.get_transferred_bytes())
                    self.endpoint.bytes_up += transferred_bytes
                    node.overlays[0].endpoint.bytes_down += transferred_bytes

                    self.logger.info("Transfer %s => %s %s at t=%f and took %f s.",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     "completed" if transfer_success else "failed",
                                     transfer_start_time, transfer_time)
                else:
                    self.endpoint.bytes_up += transfer_size
                    node.overlays[0].endpoint.bytes_down += transfer_size

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)

    async def on_receive(self, result: TransferResult):
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())

        if json_data["type"] == "chunk":
            incoming_chunk = torch.from_numpy(np.frombuffer(result.data, dtype=np.float32).copy())
            self.received_model_chunk(json_data["round"], json_data["step"], json_data["idx"], incoming_chunk)
            return
        else:
            raise RuntimeError("Received unknown message type %s" % json_data["type"])

    def received_model_chunk(self, round_nr: int, step: int, chunk_idx: int, chunk) -> None:
        if round_nr not in self.round_info:
            # We received a chunk but haven't started this round yet - store it.
            new_round = Round(round_nr)
            self.round_info[round_nr] = new_round
            new_round.out_of_order_chunks[step] = (chunk_idx, chunk)
        else:
            # Otherwise, process it right away!
            reduction_manager = self.round_info[round_nr].reduction_manager
            if reduction_manager:
                # We started the reduction process already

                # Are we waiting for this particular chunk? If so, process it right away.
                if reduction_manager.step == step:
                    self.round_info[round_nr].reduction_manager.process_received_chunk(step, chunk_idx, chunk)
                else:
                    # Otherwise, store it for processing later.
                    self.round_info[round_nr].out_of_order_chunks[step] = (chunk_idx, chunk)
            else:
                # We didn't start the reduction process yet so just store it
                self.round_info[round_nr].out_of_order_chunks[step] = (chunk_idx, chunk)
