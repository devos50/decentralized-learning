import asyncio
import json
from asyncio import ensure_future
from binascii import unhexlify
from typing import Dict, List, Optional, Tuple

import networkx as nx

from torch import Future

from accdfl.core.community import LearningCommunity
from accdfl.core.gradient_aggregation import GradientAggregation
from accdfl.core.models import serialize_adapter, unserialize_adapter
from accdfl.teleportation.sample_manager import SampleManager
from accdfl.util.eva.result import TransferResult
from pyipv8.ipv8.peer import Peer
from simulations.bandwidth_scheduler import BWScheduler


class TeleportationCommunity(LearningCommunity):
    community_id = unhexlify('e5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round: int = 0
        self.incoming_adapters: List[Tuple[bytes, Dict]] = []  # Incoming adapters for a round
        self.nodes = None
        self.bandwidth: Optional[float] = None
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []
        self.aggregator: Optional[GradientAggregation] = None
        self.node_id: int = -1

        self.bw_scheduler: BWScheduler = BWScheduler(self.my_peer.public_key.key_to_bin(),
                                                     self.peer_manager.get_my_short_id())

    def start(self):
        """
        Start to participate in the training process.
        """
        super().start()

    def go_offline(self, graceful: bool = True):
        super().go_offline(graceful=graceful)
        train_task_name = "round_%d" % self.round
        if self.is_pending_task_active(train_task_name):
            self.logger.warning("Cancelling training task of participant %s as it goes offline",
                                self.peer_manager.get_my_short_id())
            self.cancel_pending_task(train_task_name)

        self.bw_scheduler.kill_all_transfers()

    def eva_send_adapter(self, round, adapter: Dict, peer, in_sample: bool = True):
        start_time = asyncio.get_event_loop().time()
        serialized_adapter = serialize_adapter(adapter)
        response = {"round": round, "in_sample": in_sample}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_adapter(peer, serialized_response, serialized_adapter, start_time)

    def schedule_eva_send_adapter(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float) -> Future:
        # Schedule the transfer
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
                transfer_size: int = self.serialized_adapter_size + len(serialized_response)
                if self.bw_scheduler.bw_limit > 0:
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size)
                    self.logger.info("Adapter transfer %s => %s started at t=%f (size: %d)",
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

                    self.logger.info("Adapter transfer %s => %s %s at t=%f and took %f s.",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     "completed" if transfer_success else "failed",
                                     transfer_start_time, transfer_time)
                else:
                    self.endpoint.bytes_up += transfer_size
                    node.overlays[0].endpoint.bytes_down += transfer_size

                json_data = json.loads(serialized_response.decode())
                self.transfers.append((self.peer_manager.get_my_short_id(),
                                       node.overlays[0].peer_manager.get_my_short_id(), json_data["round"],
                                       transfer_start_time, transfer_time, "adapter", transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)

    def start_round(self, round_nr: int):
        self.round = round_nr
        self.register_task("round_%d" % round_nr, self.do_round)

    async def do_round(self):
        """
        Perform a single round. This method is expected to be called by a global coordinator.
        """
        self.logger.info("Peer %s starting round %d", self.peer_manager.get_my_short_id(), self.round)

        # Train
        await self.model_manager.train()

        # Detach the tensors of the adapter by making a copy
        adapter_cpy = unserialize_adapter(serialize_adapter(self.model_manager.adapter))

        my_peer_pk = self.my_peer.public_key.key_to_bin()
        self.incoming_adapters.append((my_peer_pk, adapter_cpy))

        # Send the trained adapter to your neighbours
        topology: nx.Graph = self.simulation.get_topology()
        current_sample: List[int] = SampleManager.get_sample(self.round, len(self.nodes), self.settings.teleportation.sample_size)
        my_rank: int = current_sample.index(self.node_id)

        for nb_node_id in topology.neighbors(my_rank):
            nb_peer_pk = self.nodes[current_sample[nb_node_id]].overlays[0].my_peer.public_key.key_to_bin()
            peer = self.get_peer_by_pk(nb_peer_pk)
            if not peer:
                self.logger.warning("Participant %s cannot find Peer object for participant %s!",
                                    self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(nb_peer_pk))
                continue

            self.logger.info("Participant %s sending adapter of round %d to participant %s",
                             self.peer_manager.get_my_short_id(), self.round,
                             self.peer_manager.get_short_id(peer.public_key.key_to_bin()))
            ensure_future(self.eva_send_adapter(self.round, self.model_manager.adapter, peer, in_sample=True))

    def aggregate_adapters(self):
        """
        Aggregate the received adapters.
        """
        assert self.incoming_adapters, "No incoming adapters to aggregate!"

        # The round is complete - wrap it up and proceed
        self.logger.info("Participant %s received %d adapters, aggregating...",
                         self.peer_manager.get_my_short_id(), len(self.incoming_adapters))
        self.model_manager.incoming_trained_adapters = dict((x, y) for x, y in self.incoming_adapters)
        self.model_manager.aggregator = self.aggregator

        self.model_manager.aggregate_trained_adapters()
        self.model_manager.adopt_adapter(self.model_manager.global_adapter)

        # if self.round_complete_callback:
        #     ensure_future(self.round_complete_callback(self.round))
        # if self.aggregate_complete_callback:
        #     ensure_future(self.aggregate_complete_callback(self.round, self.model_manager.adapter))
        # self.logger.info("Peer %s completed round %d", self.peer_manager.get_my_short_id(), self.round)
        self.incoming_adapters = []

    async def on_receive(self, result: TransferResult):
        """
        We received an adapter from a neighbouring peer. Store it and check if we received enough adapters to proceed.
        """
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')

        json_data = json.loads(result.info.decode())
        incoming_adapter = unserialize_adapter(result.data)
        if json_data["in_sample"]:
            self.process_incoming_adapter_from_same_sample(incoming_adapter, peer_pk)
        else:
            self.process_incoming_adapter_from_next_sample(incoming_adapter, peer_pk)

    def process_incoming_adapter_from_same_sample(self, incoming_adapter: Dict, peer_pk: bytes):
        self.incoming_adapters.append((peer_pk, incoming_adapter))

        if len(self.incoming_adapters) == self.settings.teleportation.k + 1:
            my_peer_id = self.peer_manager.get_my_short_id()
            self.logger.info("Participant %s has enough adapters to proceed", my_peer_id)
            self.aggregate_adapters()

            # Send the adapter to the nodes in the next sample
            current_sample: List[int] = SampleManager.get_sample(self.round, len(self.nodes), self.settings.teleportation.sample_size)
            my_rank: int = current_sample.index(self.node_id)
            next_sample: List[int] = SampleManager.get_sample(self.round + 1, len(self.nodes), self.settings.teleportation.sample_size)

            peer_pk = self.nodes[next_sample[my_rank]].overlays[0].my_peer.public_key.key_to_bin()
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Participant %s cannot find Peer object for participant %s!",
                                    self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))

            self.logger.info("Participant %s sending adapter of round %d to participant %s in next sample",
                             self.peer_manager.get_my_short_id(), self.round,
                             self.peer_manager.get_short_id(peer.public_key.key_to_bin()))
            ensure_future(self.eva_send_adapter(self.round, self.model_manager.adapter, peer, in_sample=False))

    def process_incoming_adapter_from_next_sample(self, incoming_adapter: Dict, peer_pk: bytes):
        self.model_manager.adopt_adapter(incoming_adapter)
        self.simulation.on_node_round_done()
