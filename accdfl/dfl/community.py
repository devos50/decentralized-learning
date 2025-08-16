import asyncio
import copy
import json
import pickle
import time
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from math import floor
from random import Random
from typing import Dict, Optional, List, Tuple, Set

from transformers import PreTrainedModel

from accdfl.core.gradient_aggregation import GradientAggregation
from accdfl.core.models import serialize_adapter, unserialize_adapter
from ipv8.lazy_community import lazy_wrapper_wd
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload, GlobalTimeDistributionPayload
from ipv8.types import Peer
from ipv8.util import succeed

from accdfl.core import NodeMembershipChange
from accdfl.core.community import LearningCommunity
from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import SessionSettings
from accdfl.dfl.caches import PingPeersRequestCache, PingRequestCache
from accdfl.dfl.payloads import AdvertiseMembership, PingPayload, PongPayload, AggAckPayload
from accdfl.dfl.sample_manager import SampleManager
from accdfl.util.eva.result import TransferResult
from simulations.bandwidth_scheduler import BWScheduler


class DFLCommunity(LearningCommunity):
    community_id = unhexlify('d5889074c1e4c60423cee6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nodes = None
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []

        self.bw_scheduler: BWScheduler = BWScheduler(self.my_peer.public_key.key_to_bin(),
                                                     self.peer_manager.get_my_short_id())

        self.random = Random(int.from_bytes(self.my_peer.public_key.key_to_bin(), 'big'))

        # Statistics
        self.active_peers_history = []
        self.bw_in_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "adapter": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
                "aggack": 0,
            },
            "num": {
                "adapter": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
                "aggack": 0,
            }
        }

        self.bw_out_stats: Dict[str, Dict[str, int]] = {
            "bytes": {
                "adapter": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
                "aggack": 0,
            },
            "num": {
                "adapter": 0,
                "view": 0,
                "ping": 0,
                "pong": 0,
                "membership": 0,
                "aggack": 0,
            }
        }
        self.determine_sample_durations = []
        self.derived_samples: List[Tuple[int, List[str]]] = []
        self.events: List[Tuple[float, str, int, str]] = []

        # State
        self.ongoing_training_task_name: Optional[str] = None
        self.train_sample_estimate: int = 0
        self.advertise_index: int = 1
        self.aggregator: Optional[GradientAggregation] = None
        self.aggregations: Dict[int, ModelManager] = {}
        self.aggregation_timeouts = set()
        self.aggregations_completed = set()
        self.completed_training = False
        self.train_future: Optional[Future] = None

        # Components
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup

        self.other_nodes_bws: Dict[bytes, int] = {}

        self.add_message_handler(AdvertiseMembership, self.on_membership_advertisement)
        self.add_message_handler(PingPayload, self.on_ping)
        self.add_message_handler(PongPayload, self.on_pong)
        self.add_message_handler(AggAckPayload, self.on_agg_ack)

    def log_event(self, round: int, event: str):
        cur_time = asyncio.get_event_loop().time()
        self.events.append((cur_time, self.peer_manager.get_my_short_id(), round, event))

    def start(self, advertise_join: bool = False):
        """
        Start to participate in the training process.
        """
        super().start()

        if advertise_join:
            self.advertise_membership(NodeMembershipChange.JOIN)

    def setup(self, settings: SessionSettings, base_model: PreTrainedModel):
        self.logger.info("Setting up experiment with %d initial participants and sample size %d (I am participant %s)" %
                         (len(settings.participants), settings.dfl.sample_size, self.peer_manager.get_my_short_id()))
        super().setup(settings, base_model)
        self.peer_manager.inactivity_threshold = settings.dfl.inactivity_threshold
        self.sample_manager = SampleManager(self.peer_manager, settings.dfl.sample_size, settings.dfl.num_aggregators)

    def get_round_estimate(self) -> int:
        """
        Get the highest round estimation, based on our local estimations and the estimations in the population view.
        """
        max_round_in_population_view = self.peer_manager.get_highest_round_in_population_view()
        max_in_aggs = max(list(self.aggregations.keys())) if self.aggregations else 0
        return max(self.train_sample_estimate, max_in_aggs, max_round_in_population_view)

    def go_online(self):
        if self.is_active:
            self.logger.warning("Participant %s already online - ignoring", self.peer_manager.get_my_short_id())
            return

        super().go_online()
        self.advertise_membership(NodeMembershipChange.JOIN)

    def go_offline(self, graceful: bool = True) -> None:
        if not self.is_active:
            self.logger.warning("Participant %s already offline - ignoring", self.peer_manager.get_my_short_id())
            return

        super().go_offline()
        self.bw_scheduler.kill_all_transfers()

        if self.aggregations:
            self.logger.warning("Aggregator %s went offline during aggregation - this might impact liveness",
                                self.peer_manager.get_my_short_id())

            for agg_round, model_manager in self.aggregations.items():
                peers_to_inform: Set[bytes] = set()

                # Get peers that are currently sending their model to this aggregator
                for transfer in self.bw_scheduler.incoming_transfers:
                    if transfer.metadata and transfer.metadata["round"] == agg_round and transfer.metadata["type"] == "aggregated_model":
                        peers_to_inform.add(transfer.sender_scheduler.peer_pk)

                # Get peers that have sent their model to this aggregator
                for peer_pk in model_manager.incoming_trained_models.keys():
                    peers_to_inform.add(peer_pk)

                # Also determine the peers in the sample size
                candidate_peers = self.sample_manager.get_ordered_sample_list(
                    agg_round - 1, self.peer_manager.get_active_peers(agg_round - 1))[:self.settings.dfl.sample_size]
                for candidate_peer in candidate_peers:
                    peers_to_inform.add(candidate_peer)

                self.logger.info("Aggregator %s will inform %d peers of failed aggregation",
                                 self.peer_manager.get_my_short_id(), len(peers_to_inform))
                for peer_pk in peers_to_inform:
                    peer = self.get_peer_by_pk(peer_pk)
                    self.send_agg_ack(peer, agg_round - 1, False)

            self.aggregations = {}
            for task_name in self.aggregation_timeouts:
                self.cancel_pending_task(task_name)
            self.aggregation_timeouts = set()

        # Cancel training
        self.cancel_current_training_task()
        self.completed_training = True
        self.train_future = None

        if graceful:
            self.advertise_membership(NodeMembershipChange.LEAVE)
        else:
            self.cancel_all_pending_tasks()

    def update_population_view_history(self):
        active_peers = self.peer_manager.get_active_peers()
        active_peers = [self.peer_manager.get_short_id(peer_pk) for peer_pk in active_peers]

        if not self.active_peers_history or (self.active_peers_history[-1][1] != active_peers):  # It's the first entry or it has changed
            self.active_peers_history.append((time.time(), active_peers))

    def advertise_membership(self, change: NodeMembershipChange):
        """
        Advertise your (new) membership to random (online) peers.
        """
        advertise_index: int = self.advertise_index
        self.advertise_index += 1

        self.logger.info("Participant %s advertising its membership change %s to active participants (idx %d)",
                         self.peer_manager.get_my_short_id(), change, advertise_index)

        active_peer_pks = self.peer_manager.get_active_peers()
        if self.my_id in active_peer_pks:
            active_peer_pks.remove(self.my_id)

        if change == NodeMembershipChange.LEAVE:
            # When going offline, we can simply query our current view of the network and select the last nodes offline
            random_peer_pks = self.random.sample(active_peer_pks, min(self.sample_manager.sample_size * 10, len(active_peer_pks)))
        else:
            # When coming online we probably don't have a fresh view on the network so we need to determine online nodes
            peer_pks = self.peer_manager.get_peers()
            random_peer_pks = self.random.sample(peer_pks, min(self.sample_manager.sample_size * 10, len(peer_pks)))

        if self.advertise_index > (advertise_index + 1):
            # It's not relevant anymore what we're doing
            return

        for peer_pk in random_peer_pks:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Cannot find Peer object for participant %s!",
                                    self.peer_manager.get_short_id(peer_pk))
            self.logger.debug("Participant %s advertising its membership change to participant %s",
                              self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
            global_time = self.claim_global_time()
            auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
            payload = AdvertiseMembership(self.get_round_estimate(), advertise_index, change.value)
            dist = GlobalTimeDistributionPayload(global_time)
            packet = self._ez_pack(self._prefix, AdvertiseMembership.msg_id, [auth, dist, payload])
            self.bw_out_stats["bytes"]["membership"] += len(packet)
            self.bw_out_stats["num"]["membership"] += 1
            self.endpoint.send(peer.address, packet)

        # Update your own population view
        info = self.peer_manager.last_active[self.my_id]
        self.peer_manager.last_active[self.my_id] = (info[0], (advertise_index, change))

    @lazy_wrapper_wd(GlobalTimeDistributionPayload, AdvertiseMembership)
    def on_membership_advertisement(self, peer, dist, payload, raw_data: bytes):
        """
        We received a membership advertisement from a new peer.
        """
        if not self.is_active:
            return

        self.bw_in_stats["bytes"]["membership"] += len(raw_data)
        self.bw_in_stats["num"]["membership"] += 1

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        change: NodeMembershipChange = NodeMembershipChange(payload.change)
        latest_round = self.get_round_estimate()
        if change == NodeMembershipChange.JOIN:
            self.logger.debug("Participant %s updating membership of participant %s to: JOIN (idx %d)",
                              self.peer_manager.get_my_short_id(), peer_id, payload.index)
            # Do not apply this immediately since we do not want the newly joined node to be part of the next sample just yet.
            self.peer_manager.last_active_pending[peer_pk] = (
            max(payload.round, latest_round), (payload.index, NodeMembershipChange.JOIN))
        else:
            self.logger.debug("Participant %s updating membership of participant %s to: LEAVE (idx %d)",
                              self.peer_manager.get_my_short_id(), peer_id, payload.index)
            self.peer_manager.last_active[peer_pk] = (
            max(payload.round, latest_round), (payload.index, NodeMembershipChange.LEAVE))

    def determine_available_peers_for_sample(self, sample: int, count: int,
                                             getting_aggregators: bool = False, pick_active_peers: bool = True) -> Future:
        if getting_aggregators and self.settings.dfl.fixed_aggregator:
            candidate_peers = [self.settings.dfl.fixed_aggregator]
        else:
            if pick_active_peers:
                raw_peers = self.peer_manager.get_active_peers(sample)
            else:
                raw_peers = self.peer_manager.get_peers()
            candidate_peers = self.sample_manager.get_ordered_sample_list(sample, raw_peers)
        self.logger.info("Participant %s starts to determine %d available peers in sample %d (candidates: %d)",
                         self.peer_manager.get_my_short_id(), count, sample,
                         len(candidate_peers))

        if getting_aggregators and not self.settings.dfl.fixed_aggregator and self.other_nodes_bws:
            # Filter the candidates in the sample and sort them based on their bandwidth capabilities
            candidate_peers = sorted(candidate_peers[:self.settings.dfl.sample_size],
                                     key=lambda pk: self.other_nodes_bws[pk], reverse=True)

        cache = PingPeersRequestCache(self, candidate_peers, count, sample)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def ping_peer(self, ping_all_id: int, peer_pk: bytes) -> Future:
        self.logger.debug("Participant %s pinging participant %s",
                          self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
        peer_short_id = self.peer_manager.get_short_id(peer_pk)
        peer = self.get_peer_by_pk(peer_pk)
        if not peer:
            self.logger.warning("Wanted to ping participant %s but cannot find Peer object!", peer_short_id)
            return succeed((peer_pk, False))

        cache = PingRequestCache(self, ping_all_id, peer, self.settings.dfl.ping_timeout)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def send_ping(self, peer: Peer, identifier: int) -> None:
        """
        Send a ping message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PingPayload(self.get_round_estimate(), self.advertise_index - 1, identifier)

        packet = self._ez_pack(self._prefix, PingPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["ping"] += len(packet)
        self.bw_out_stats["num"]["ping"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(PingPayload)
    def on_ping(self, peer: Peer, payload: PingPayload, raw_data: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring ping message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.bw_in_stats["bytes"]["ping"] += len(raw_data)
        self.bw_in_stats["num"]["ping"] += 1

        if peer_pk in self.peer_manager.last_active:
            self.peer_manager.update_peer_activity(peer_pk, max(self.get_round_estimate(), payload.round))
            if payload.index > self.peer_manager.last_active[peer_pk][1][0]:
                self.peer_manager.last_active[peer_pk] = (self.peer_manager.last_active[peer_pk][0],
                                                          (payload.index, NodeMembershipChange.JOIN))

        self.send_pong(peer, payload.identifier)

    def send_pong(self, peer: Peer, identifier: int) -> None:
        """
        Send a pong message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PongPayload(self.get_round_estimate(), self.advertise_index - 1, identifier)

        packet = self._ez_pack(self._prefix, PongPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["pong"] += len(packet)
        self.bw_out_stats["num"]["pong"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(PongPayload)
    def on_pong(self, peer: Peer, payload: PongPayload, raw_data: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        my_peer_id = self.peer_manager.get_my_short_id()
        peer_short_id = self.peer_manager.get_short_id(peer_pk)

        if not self.is_active:
            self.logger.debug("Participant %s ignoring ping message from %s due to inactivity",
                              my_peer_id, peer_short_id)
            return

        self.logger.debug("Participant %s receiving pong message from participant %s", my_peer_id, peer_short_id)

        self.bw_in_stats["bytes"]["pong"] += len(raw_data)
        self.bw_in_stats["num"]["pong"] += 1

        if not self.request_cache.has("ping-%s" % peer_short_id, payload.identifier):
            self.logger.warning("ping cache with id %s not found", payload.identifier)
            return

        if peer_pk in self.peer_manager.last_active:
            self.peer_manager.update_peer_activity(peer_pk, max(self.get_round_estimate(), payload.round))
            if payload.index > self.peer_manager.last_active[peer_pk][1][0]:
                self.peer_manager.last_active[peer_pk] = (self.peer_manager.last_active[peer_pk][0],
                                                          (payload.index, NodeMembershipChange.JOIN))

        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(),
                                               max(self.get_round_estimate(), payload.round))

        cache = self.request_cache.pop("ping-%s" % peer_short_id, payload.identifier)
        cache.on_pong()

    def send_agg_ack(self, peer: Peer, round: int, success: bool) -> None:
        """
        Send a ping message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = AggAckPayload(round, success)

        packet = self._ez_pack(self._prefix, AggAckPayload.msg_id, [auth, payload])
        self.bw_out_stats["bytes"]["aggack"] += len(packet)
        self.bw_out_stats["num"]["aggack"] += 1
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper_wd(AggAckPayload)
    def on_agg_ack(self, peer: Peer, payload: AggAckPayload, raw_data: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        my_peer_id = self.peer_manager.get_my_short_id()
        peer_short_id = self.peer_manager.get_short_id(peer_pk)

        if not self.is_active:
            self.logger.debug("Participant %s ignoring ping message from %s due to inactivity",
                              my_peer_id, peer_short_id)
            return

        self.logger.info("Participant %s receiving agg ack message from aggregator %s for round %d", my_peer_id, peer_short_id, payload.round)

        self.bw_in_stats["bytes"]["aggack"] += len(raw_data)
        self.bw_in_stats["num"]["aggack"] += 1

        # We can safely wrap up training in this round.
        if payload.success:
            if self.train_future and self.train_sample_estimate == payload.round:
                self.train_future.set_result(True)
            else:
                self.logger.warning("Participant %s ignoring agg ack as it's not training or the incoming agg ack is for an invalid round", my_peer_id)
        else:
            # Mark this aggregator as offline
            self.peer_manager.last_active[peer_pk] = (payload.round, (0, NodeMembershipChange.LEAVE))

            # Try to send the model to the next eligible aggregator
            if self.train_future:
                ensure_future(self.forward_trained_model(payload.round))

    def train_in_round(self, round):
        self.ongoing_training_task_name = "round_%d" % round
        if not self.is_pending_task_active(self.ongoing_training_task_name):
            task = self.register_task(self.ongoing_training_task_name, self.train_in_round_coroutine, round)

    async def train_in_round_coroutine(self, round):
        """
        Participate in a round.
        """
        if round < 1:
            raise RuntimeError("Round number %d invalid!" % round)

        self.logger.info("Participant %s starts participating in round %d", self.peer_manager.get_my_short_id(), round)
        self.completed_training = False
        self.log_event(round, "start_train")

        # 1. Train the model
        await self.model_manager.train()

        self.log_event(round, "done_train")

        # It might be that we went offline at this point - check for it
        if not self.is_active:
            self.logger.warning("Participant %s went offline during model training in round %d - not proceeding", self.peer_manager.get_my_short_id(), round)
            return

        await self.forward_trained_adapter(round)

    async def forward_trained_adapter(self, round: int):
        # 2. Determine the aggregators of the next sample that are available
        aggregators = await self.determine_available_peers_for_sample(round + 1, self.settings.dfl.num_aggregators,
                                                                      getting_aggregators=True)
        aggregator_ids: List[str] = [self.peer_manager.get_short_id(peer_id) for peer_id in aggregators]
        self.derived_samples.append((round + 1, aggregator_ids))
        self.logger.info("Participant %s determined %d available aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), len(aggregator_ids), round + 1, aggregator_ids)

        # 3. Send the trained adapter to the aggregators in the next sample
        self.train_future = Future()
        await self.send_trained_adapter_to_aggregators(aggregators, round + 1)

        if self.train_future:  # Could be interrupted
            await self.train_future

            self.completed_training = True
            self.ongoing_training_task_name = None
            self.train_future = None
            self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round)
            if self.round_complete_callback:
                ensure_future(self.round_complete_callback(round))

    async def send_aggregated_adapter_to_participants(self, participants: List[bytes], sample_index: int) -> List[bool]:
        if not self.is_active:
            self.logger.warning("Participant %s not sending aggregated adapter due to offline status",
                                self.peer_manager.get_my_short_id())
            return []

        self.logger.info("Participant %s sending aggregated adapter of round %d to participants",
                         self.peer_manager.get_my_short_id(), sample_index - 1)

        # For load balancing purposes, shuffle this list
        self.random.shuffle(participants)

        futures: List[Future] = []
        population_view = copy.deepcopy(self.peer_manager.last_active)
        for peer_pk in participants:
            if peer_pk == self.my_id:
                asyncio.get_event_loop().call_soon(self.received_aggregated_adapter, self.my_peer, sample_index, self.model_manager.global_adapter)
                continue

            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue

            futures.append(self.eva_send_adapter(sample_index, "aggregated_adapter", population_view, peer, self.model_manager.global_adapter))

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

        res = await asyncio.gather(*futures)
        return res

    async def send_trained_adapter_to_aggregators(self, aggregators: List[bytes], sample_index: int) -> None:
        """
        Send the current adapter to the aggregators in a particular sample.
        """
        if not self.is_active:
            self.logger.warning("Participant %s not sending trained adapter due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        aggregator_ids = [self.peer_manager.get_short_id(aggregator) for aggregator in aggregators]
        self.logger.info("Participant %s sending trained adapter of round %d to %d aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), sample_index - 1, len(aggregators), sample_index, aggregator_ids)
        population_view = copy.deepcopy(self.peer_manager.last_active)

        # For load balancing purposes, shuffle this list
        self.random.shuffle(aggregators)

        futures: List[Future] = []
        for aggregator in aggregators:
            if aggregator == self.my_id:
                self.logger.info("Participant %s sending trained adapter to self", self.peer_manager.get_my_short_id())
                ensure_future(self.received_trained_adapter(self.my_peer, sample_index, self.model_manager.adapter))
                continue

            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue

            futures.append(self.eva_send_adapter(sample_index, "trained_adapter", population_view, peer, self.model_manager.adapter))

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

        await asyncio.gather(*futures)

    async def eva_send_adapter(self, round, type, population_view, peer, adapter: Dict):
        serialized_adapter = serialize_adapter(adapter)
        serialized_population_view = pickle.dumps(population_view)
        # TODO When using FedAdam, we need to count the momentum states as well
        self.bw_out_stats["bytes"]["adapter"] += self.serialized_adapter_size
        self.bw_out_stats["bytes"]["view"] += len(serialized_population_view)
        self.bw_out_stats["num"]["adapter"] += 1
        self.bw_out_stats["num"]["view"] += 1
        binary_data = serialized_adapter + serialized_population_view
        response = {"round": round, "type": type, "adapter_data_len": len(serialized_adapter)}
        serialized_response = json.dumps(response).encode()

        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                transfer_start_time = asyncio.get_event_loop().time()
                transfer_size: int = len(serialized_population_view) + self.serialized_adapter_size + len(serialized_response)
                if self.bw_scheduler.bw_limit > 0:
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size)
                    transfer.metadata = response
                    self.logger.info("Adapter transfer %s => %s started at t=%f (size: %d)",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time, transfer_size)
                    try:
                        await transfer.complete_future
                    except RuntimeError:
                        transfer_success = False
                    transfer_time = asyncio.get_event_loop().time() - transfer_start_time

                    transferred_bytes: int = transfer.get_transferred_bytes()
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
                                       transfer_start_time, transfer_time, json_data["type"], transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)

        return transfer_success

    def cancel_current_training_task(self):
        if self.ongoing_training_task_name and self.is_pending_task_active(self.ongoing_training_task_name):
            self.logger.info("Participant %s interrupting training task %s",
                             self.peer_manager.get_my_short_id(), self.ongoing_training_task_name)
            self.cancel_pending_task(self.ongoing_training_task_name)
        self.ongoing_training_task_name = None

    async def on_receive(self, result: TransferResult):
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())
        serialized_adapter = result.data[:json_data["adapter_data_len"]]
        serialized_population_view = result.data[json_data["adapter_data_len"]:]
        received_population_view = pickle.loads(serialized_population_view)
        self.bw_in_stats["bytes"]["adapter"] += len(serialized_adapter)
        self.bw_in_stats["bytes"]["view"] += len(serialized_population_view)
        self.bw_in_stats["num"]["adapter"] += 1
        self.bw_in_stats["num"]["view"] += 1
        self.peer_manager.merge_population_views(received_population_view)
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(),
                                               max(json_data["round"], self.get_round_estimate()))
        incoming_adapter = unserialize_adapter(serialized_adapter)

        if json_data["type"] == "trained_adapter":
            self.log_event(json_data["round"], "received_trained_adapter")
            await self.received_trained_adapter(result.peer, json_data["round"], incoming_adapter)
        elif json_data["type"] == "aggregated_adapter":
            self.log_event(json_data["round"], "received_aggregated_adapter")
            self.received_aggregated_adapter(result.peer, json_data["round"], incoming_adapter)

    def has_enough_trained_adapters(self, agg_round: int) -> bool:
        return len(self.aggregations[agg_round].incoming_trained_adapters) >= \
               floor(self.settings.dfl.sample_size * self.settings.dfl.success_fraction)

    def has_enough_trained_adapters_for_liveness(self, agg_round: int) -> bool:
        return len(self.aggregations[agg_round].incoming_trained_adapters) >= 3

    async def received_trained_adapter(self, peer: Peer, index: int, adapter: Dict) -> None:
        model_round = index - 1  # The round associated with this model is one smaller than the sample index
        if self.shutting_down:
            self.logger.warning("Participant %s ignoring incoming trained adapter due to shutdown",
                                self.peer_manager.get_my_short_id())
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received trained adapter for round %d from participant %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        if index not in self.aggregations:
            if index in self.aggregations_completed:
                self.logger.info("Participant %s received trained adapter for completed round %d - ignoring ",
                                 self.peer_manager.get_my_short_id(), model_round)
                return

            self.logger.info("Participant %s received trained adapter for round %d for the first time - "
                             "starting to aggregate", self.peer_manager.get_my_short_id(), model_round)
            self.log_event(model_round, "start_aggregate")

            # Set the round timeout
            if self.settings.dfl.aggregation_timeout > 0:
                task_name = "aggregate_%d_timeout" % model_round
                self.register_task(task_name, self.on_aggregation_timeout, model_round,
                                   index, delay=self.settings.dfl.aggregation_timeout)
                self.aggregation_timeouts.add(task_name)

            model_manager = ModelManager(self.model_manager.peft_model, self.settings, self.model_manager.participant_index)
            model_manager.global_adapter = self.model_manager.global_adapter
            model_manager.aggregator = self.aggregator
            self.aggregations[index] = model_manager

        if index not in self.aggregations_completed:
            self.aggregations[index].process_incoming_trained_adapter(peer_pk, adapter)

            # Check whether we received enough incoming adapters
            if self.has_enough_trained_adapters(index):
                self.logger.info("Aggregator %s received sufficient trained adapters (%d) of round %d",
                                 self.peer_manager.get_my_short_id(), len(self.aggregations[index].incoming_trained_adapters),
                                 model_round)
                await self.aggregator_complete_round(model_round, index)
            else:
                self.logger.info("Aggregator %s has not enough trained adapters (%d) of round %d yet",
                                 self.peer_manager.get_my_short_id(), len(self.aggregations[index].incoming_trained_adapters),
                                 model_round)

    async def aggregator_complete_round(self, model_round: int, index: int):
        model_manager = self.aggregations[index]
        self.aggregations_completed.add(index)

        # Stop the timeout task
        timeout_task_name: str = "aggregate_%d_timeout" % model_round
        if self.is_pending_task_active(timeout_task_name):
            self.cancel_pending_task(timeout_task_name)
        self.aggregation_timeouts.remove(timeout_task_name)

        # 3.1. Aggregate these adapters
        self.logger.info("Aggregator %s will average the adapters of round %d",
                         self.peer_manager.get_my_short_id(), model_round)
        model_manager.aggregate_trained_adapters()

        if self.aggregate_complete_callback:
            ensure_future(self.aggregate_complete_callback(model_round))

        # Capture the peers
        peers_that_sent_trained_model: List[bytes] = list(model_manager.incoming_trained_adapters.keys())

        # 3. Determine the participants of the next sample that are available
        participants = await self.determine_available_peers_for_sample(index, self.settings.dfl.sample_size)
        participants_ids: List[str] = [self.peer_manager.get_short_id(peer_id) for peer_id in participants]
        self.derived_samples.append((index, participants_ids))
        self.logger.info("Participant %s determined %d available participants for round %d: %s",
                         self.peer_manager.get_my_short_id(), len(participants_ids), model_round, participants_ids)

        # 3.3. Distribute the average model to the available participants in the sample.
        await self.send_aggregated_adapter_to_participants(participants, index)

        if not self.is_active:
            # It might be that the aggregator went offline
            self.logger.warning("Aggregator %s went offline - not continuing", self.peer_manager.get_my_short_id())
            return

        # 3.4. Send acknowledgement to the previous sample that aggregation has completed.
        for peer_pk in peers_that_sent_trained_model:
            peer = self.get_peer_by_pk(peer_pk)
            self.send_agg_ack(peer, model_round, True)

        # 4. Invoke the complete callback
        self.logger.info("Aggregator %s completed aggregation and sending in round %d",
                         self.peer_manager.get_my_short_id(), model_round)

        if index in self.aggregations:
            self.aggregations.pop(index)

        self.log_event(model_round, "done_aggregation")

    def on_aggregation_timeout(self, model_round: int, index: int):
        self.logger.info("Aggregator %s triggered aggregation timeout in round %d - wrapping up",
                         self.peer_manager.get_my_short_id(), model_round)

        if index not in self.aggregations:
            return  # Just ignore it, the aggregator might have gone offline

        if self.aggregations_completed and max(self.aggregations_completed) > index:
            self.logger.warning("Timeout triggered for aggregator %s but work is irrelevant as we already completed "
                                "aggregation for a subsequent round, ignoring it", self.peer_manager.get_my_short_id())
            return

        if self.has_enough_trained_adapters_for_liveness(index):
            self.log_event(model_round, "aggregate_timeout")
            ensure_future(self.aggregator_complete_round(model_round, index))
        else:
            self.logger.info("Aggregator %s triggered aggregation timeout in round %d but didn't receive sufficient "
                             "adapters to continue (%d adapters received)",
                             self.peer_manager.get_my_short_id(), model_round,
                             len(self.aggregations[index].incoming_trained_adapters))
            self.aggregations.pop(index)

    def received_aggregated_adapter(self, peer: Peer, model_round: int, aggregated_adapter: Dict) -> None:
        if self.shutting_down:
            self.logger.warning("Participant %s ignoring incoming aggregated adapter due to shutdown",
                                self.peer_manager.get_my_short_id())
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received aggregated adapter of round %d from aggregator %s",
                         self.peer_manager.get_my_short_id(), model_round - 1, peer_id)

        if model_round > self.train_sample_estimate:
            if model_round > 1:  # We don't want to log this for the first round
                self.log_event(self.train_sample_estimate, "interrupt_training")
            self.train_sample_estimate = model_round
            self.cancel_current_training_task()
            self.completed_training = False
        if model_round == self.train_sample_estimate and not self.ongoing_training_task_name and not self.completed_training:
            self.model_manager.adopt_adapter(aggregated_adapter)
            self.train_in_round(model_round)
        else:
            self.logger.info("Participant %s NOT starting training round %d (train sample: %d, ongoing train task: %s, "
                             "completed training: %d)", self.peer_manager.get_my_short_id(), model_round,
                             self.train_sample_estimate, self.ongoing_training_task_name, self.completed_training)

    async def on_send_complete(self, result: TransferResult):
        await super().on_send_complete(result)
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(), self.get_round_estimate())
