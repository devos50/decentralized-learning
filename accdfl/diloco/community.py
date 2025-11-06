from asyncio import Future, ensure_future
from binascii import hexlify, unhexlify
from typing import Dict, List, Optional, Tuple
from accdfl.core.community import LearningCommunity
from accdfl.core.gradient_aggregation import GradientAggregation
from accdfl.diloco.reduction_manager import ReductionManager
from accdfl.diloco.round import Round
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
        self.aggregator: Optional[GradientAggregation] = None

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

        # Train
        await self.model_manager.train()

        # 2. Share the model chunks in a ring all-reduce fashion
        my_rank = await self.do_ring_allreduce(self.round_info[self.round])
    
    async def do_ring_allreduce(self, round_info: Round) -> int:
        round_nr: int = round_info.round_nr
        participants = await self.determine_available_peers_for_sample(round_nr, self.settings.dfl.sample_size)
        participants = sorted(participants)
        total_participants: int = len(participants)
        my_rank: int = participants.index(self.my_id)

        round_info.reduction_manager = ReductionManager(round, self.model_manager.model, participants, my_rank)
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

        return my_rank
