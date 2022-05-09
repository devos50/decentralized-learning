import asyncio
import copy
import json
import pickle
import random
import time
from asyncio import Future, ensure_future, Task
from binascii import unhexlify, hexlify
from typing import Optional, Dict, List, Callable

from torch import nn

from accdfl.core import TransmissionMethod, NodeMembershipChange, State
from accdfl.core.caches import PingPeersRequestCache, PingRequestCache
from accdfl.core.exceptions import StopAggregationException
from accdfl.core.model import serialize_model, unserialize_model, create_model
from accdfl.core.model_manager import ModelManager
from accdfl.core.payloads import AdvertiseMembership, PingPayload, PongPayload
from accdfl.core.peer_manager import PeerManager
from accdfl.core.sample_manager import SampleManager
from accdfl.util.eva.protocol import EVAProtocol
from accdfl.util.eva.result import TransferResult

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload, GlobalTimeDistributionPayload
from ipv8.requestcache import RequestCache
from ipv8.types import Peer
from ipv8.util import succeed


class DFLCommunity(Community):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_cache = RequestCache()
        self.my_id = self.my_peer.public_key.key_to_bin()
        self.round_complete_callback: Optional[Callable] = None
        self.aggregate_complete_callback: Optional[Callable] = None

        # Statistics
        self.active_peers_history = []
        self.aggregation_durations: Dict[int, float] = {}

        # Settings
        self.parameters = None
        self.model_send_delay = 1.0
        self.fixed_aggregator = None
        self.train_in_subprocess = True

        # State
        self.is_active = False
        self.did_setup = False
        self.shutting_down = False
        self.ongoing_training_task_name: Optional[str] = None
        self.ongoing_aggregation_task_name: Optional[str] = None
        self.sample_index_estimate: int = 0
        self.aggregation_future: Optional[Future] = None

        # Components
        self.peer_manager: PeerManager = PeerManager(self.my_id, -1)
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup
        self.model_manager: Optional[ModelManager] = None    # Initialized when the process is setup

        # Model exchange parameters
        self.eva = EVAProtocol(self, self.on_receive, self.on_send_complete, self.on_error)
        self.data_dir = None
        self.transmission_method = TransmissionMethod.EVA
        self.eva_max_retry_attempts = 20
        self.transfer_times = []

        self.add_message_handler(AdvertiseMembership, self.on_membership_advertisement)
        self.add_message_handler(PingPayload, self.on_ping)
        self.add_message_handler(PongPayload, self.on_pong)

        self.logger.info("The DFL community started with peer ID: %s", self.peer_manager.get_my_short_id())

    def start(self, advertise_join: bool = False):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"

        self.is_active = True

        if advertise_join:
            self.advertise_membership(NodeMembershipChange.JOIN)

        # Start the process
        if self.sample_manager.is_participant_in_round(self.my_id, 1):
            self.received_aggregated_model(self.my_peer, 1, self.model_manager.model)
        else:
            self.logger.info("Participant %s won't participate in round 1", self.peer_manager.get_my_short_id())

    def setup(self, parameters: Dict, data_dir: str, transmission_method: TransmissionMethod = TransmissionMethod.EVA,
              aggregator: Optional[bytes] = None):
        if parameters["data_distribution"] == "iid":
            assert parameters["target_participants"] * parameters["local_classes"] == sum(parameters["nodes_per_class"])
        else:
            assert parameters["target_participants"] * parameters["local_shards"] * parameters["shard_size"] == sum(parameters["samples_per_class"])

        self.parameters = parameters
        self.data_dir = data_dir
        self.fixed_aggregator = aggregator
        self.logger.info("Setting up experiment with %d initial participants and sample size %d (I am participant %s)" %
                         (len(parameters["participants"]), parameters["sample_size"],
                          self.peer_manager.get_my_short_id()))

        self.peer_manager.inactivity_threshold = parameters["inactivity_threshold"]
        for participant in parameters["participants"]:
            self.peer_manager.add_peer(unhexlify(participant))
        self.sample_manager = SampleManager(self.peer_manager, parameters["sample_size"], parameters["num_aggregators"])

        # Initialize the model
        model = create_model(parameters["dataset"], parameters["model"])
        participant_index = parameters["all_participants"].index(hexlify(self.my_id).decode())
        self.model_manager = ModelManager(model, parameters, participant_index)

        # Setup the model transmission
        self.transmission_method = transmission_method
        if self.transmission_method == TransmissionMethod.EVA:
            self.logger.info("Setting up EVA protocol")
            self.eva.settings.block_size = 60000
            self.eva.settings.window_size = 64
            self.eva.settings.retransmit_attempt_count = 10
            self.eva.settings.retransmit_interval_in_sec = 1
            self.eva.settings.timeout_interval_in_sec = 10

        self.update_population_view_history()

        self.did_setup = True

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    def go_offline(self, graceful: bool = True) -> None:
        self.is_active = False

        self.logger.info("Participant %s will go offline", self.peer_manager.get_my_short_id())

        if graceful:
            info = self.peer_manager.last_active[self.my_id]
            self.peer_manager.last_active[self.my_id] = (info[0], (self.sample_index_estimate, NodeMembershipChange.LEAVE))
            self.advertise_membership(NodeMembershipChange.LEAVE)
        else:
            self.cancel_all_pending_tasks()

    def update_population_view_history(self):
        num_active_peers = self.peer_manager.get_num_peers(self.sample_index_estimate)
        if not self.active_peers_history or (self.active_peers_history[-1][1] != num_active_peers):  # It's the first entry or it has changed
            self.active_peers_history.append((time.time(), num_active_peers))

    def advertise_membership(self, change: NodeMembershipChange):
        """
        Advertise your (new) membership to random peers.
        """
        active_peer_pks = self.peer_manager.get_active_peers()
        if self.my_id in active_peer_pks:
            active_peer_pks.remove(self.my_id)

        random_peer_pks = random.sample(active_peer_pks, min(self.sample_manager.sample_size, len(active_peer_pks)))
        for peer_pk in random_peer_pks:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Cannot find Peer object for participant %s!",
                                    self.peer_manager.get_short_id(peer_pk))
            self.logger.info("Participant %s advertising its membership change to participant %s",
                              self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
            global_time = self.claim_global_time()
            auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
            payload = AdvertiseMembership(self.sample_index_estimate, change.value)
            dist = GlobalTimeDistributionPayload(global_time)
            packet = self._ez_pack(self._prefix, AdvertiseMembership.msg_id, [auth, dist, payload])
            self.endpoint.send(peer.address, packet)

    @lazy_wrapper(GlobalTimeDistributionPayload, AdvertiseMembership)
    def on_membership_advertisement(self, peer, dist, payload):
        """
        We received a membership advertisement from a new peer.
        """
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        self.logger.info("Participant %s updating membership of participant %s",
                         self.peer_manager.get_my_short_id(), peer_id)

        change: NodeMembershipChange = NodeMembershipChange(payload.change)
        latest_round = self.sample_index_estimate
        if change == NodeMembershipChange.JOIN:
            # Do not apply this immediately since we do not want the newly joined node to be part of the next sample just yet.
            self.peer_manager.last_active_pending[peer_pk] = (max(payload.round, latest_round), (payload.round, NodeMembershipChange.JOIN))
        else:
            self.peer_manager.last_active[peer_pk] = (max(payload.round, latest_round), (payload.round, NodeMembershipChange.LEAVE))

    def determine_available_aggregators_for_round(self, round: int) -> Future:
        if self.fixed_aggregator:
            candidate_aggregators = [self.fixed_aggregator]
        else:
            candidate_aggregators = self.sample_manager.get_ordered_sample_list(
                round, self.peer_manager.get_active_peers(round))
        self.logger.info("Participant %s starts to determine %d available aggregators for round %d (candidates: %d)",
                         self.peer_manager.get_my_short_id(), self.parameters["num_aggregators"], round,
                         len(candidate_aggregators))
        cache = PingPeersRequestCache(self, candidate_aggregators, self.parameters["num_aggregators"], round)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def determine_available_participants_for_round(self, round: int) -> Future:
        candidate_participants = self.sample_manager.get_ordered_sample_list(round, self.peer_manager.get_active_peers(round))
        candidate_participants_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in candidate_participants]
        self.logger.info("Aggregator %s starts to determine %d available participants for round %d (candidates: %d)",
                         self.peer_manager.get_my_short_id(), self.parameters["sample_size"], round,
                         len(candidate_participants))
        self.logger.debug("Candidates for participating in round %d: %s", round, candidate_participants_ids)
        cache = PingPeersRequestCache(self, candidate_participants, self.parameters["sample_size"], round)
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def ping_peer(self, ping_all_id: int, peer_pk: bytes, round: int) -> Future:
        self.logger.debug("Participant %s pinging participant %s for round %d",
                          self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk), round)
        peer_short_id = self.peer_manager.get_short_id(peer_pk)
        peer = self.get_peer_by_pk(peer_pk)
        if not peer:
            self.logger.warning("Wanted to ping participant %s but cannot find Peer object!", peer_short_id)
            return succeed((peer_pk, False))

        cache = PingRequestCache(self, ping_all_id, peer, round, self.parameters["ping_timeout"])
        self.request_cache.add(cache)
        cache.start()
        return cache.future

    def send_ping(self, peer: Peer, round: int, identifier: int) -> None:
        """
        Send a ping message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PingPayload(round, identifier)

        packet = self._ez_pack(self._prefix, PingPayload.msg_id, [auth, payload])
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper(PingPayload)
    def on_ping(self, peer: Peer, payload: PingPayload) -> None:
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.warning("Participant %s ignoring ping message from %s due to inactivity", my_peer_id, peer_id)
            return

        peer_pk = peer.public_key.key_to_bin()
        if peer_pk in self.peer_manager.last_active:
            self.peer_manager.update_peer_activity(peer_pk, max(self.sample_index_estimate, payload.round))

        self.send_pong(peer, payload.round, payload.identifier)

    def send_pong(self, peer: Peer, round: int, identifier: int) -> None:
        """
        Send a pong message with an identifier to a specific peer.
        """
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = PongPayload(round, identifier)

        packet = self._ez_pack(self._prefix, PongPayload.msg_id, [auth, payload])
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper(PongPayload)
    def on_pong(self, peer: Peer, payload: PongPayload) -> None:
        my_peer_id = self.peer_manager.get_my_short_id()
        peer_short_id = self.peer_manager.get_short_id(peer.public_key.key_to_bin())

        if not self.is_active:
            self.logger.warning("Participant %s ignoring ping message from %s due to inactivity",
                                my_peer_id, peer_short_id)
            return

        if not self.request_cache.has("ping-%s" % peer_short_id, payload.identifier):
            self.logger.warning("ping cache with id %s not found", payload.identifier)
            return

        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(),
                                               max(self.sample_index_estimate, payload.round))

        cache = self.request_cache.pop("ping-%s" % peer_short_id, payload.identifier)
        cache.on_pong(payload.round)

    def train_in_round(self, round):
        self.ongoing_training_task_name = "round_%d" % round
        task = self.register_task(self.ongoing_training_task_name, self.train_in_round_coroutine, round)
        task.add_done_callback(lambda f, r=round: self.on_train_completed(f, r))

    def on_train_completed(self, _, round):
        self.ongoing_training_task_name = None
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round)
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(round))

    async def train_in_round_coroutine(self, round):
        """
        Participate in a round.
        """
        if round < 1:
            raise RuntimeError("Round number %d invalid!" % round)

        self.logger.info("Participant %s starts participating in round %d", self.peer_manager.get_my_short_id(), round)

        # 1. Train the model
        await self.model_manager.train(self.train_in_subprocess)

        # 2. Determine the aggregators of the next sample that are available
        aggregators = await self.determine_available_aggregators_for_round(round + 1)
        aggregator_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in aggregators]
        self.logger.info("Participant %s determined %d available aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), len(aggregator_ids), round + 1, aggregator_ids)

        # 3. Send the model to the aggregators in the next sample
        await self.send_trained_model_to_aggregators(aggregators, round + 1)

        # 4. As last step, send the model to yourself. We want this task to be finished before doing so.
        if self.my_id in aggregators:
            asyncio.get_event_loop().call_soon(self.received_trained_model, self.my_peer, round + 1, self.model_manager.model)

    def aggregate_in_round(self, round):
        self.ongoing_aggregation_task_name = "aggregate_%d" % round
        task = self.register_task(self.ongoing_aggregation_task_name, self.aggregate_in_round_coroutine, round)
        task.add_done_callback(lambda f, r=round: self.on_aggregate_complete(f, r))

    def on_aggregate_complete(self, f, round):
        self.ongoing_aggregation_task_name = None
        self.logger.info("Aggregator %s completed aggregation in round %d", self.peer_manager.get_my_short_id(), round)
        if self.aggregate_complete_callback:
            ensure_future(self.aggregate_complete_callback(round, f.result()))

    async def aggregate_in_round_coroutine(self, round: int):
        self.logger.info("Aggregator %s starts aggregating in round %d", self.peer_manager.get_my_short_id(), round)
        start_time = time.time()

        if not self.model_manager.has_enough_trained_models():
            self.logger.info("Aggregator %s starts to wait for trained models of round %d",
                             self.peer_manager.get_my_short_id(), round)
            self.aggregation_future = Future()
            received_sufficient_models = False
            try:
                await asyncio.wait_for(self.aggregation_future, timeout=self.parameters["aggregation_timeout"])
                received_sufficient_models = True
            except asyncio.exceptions.TimeoutError:
                self.logger.warning("Aggregator %s triggered timeout while waiting for models of round %d",
                                    self.peer_manager.get_my_short_id(), round)
            except StopAggregationException:
                self.logger.warning("Aggregator %s triggered StopAggregationException while waiting for models of "
                                    "round %d", self.peer_manager.get_my_short_id(), round)
            self.aggregation_future = None
        else:
            received_sufficient_models = True

        if not received_sufficient_models:
            self.model_manager.reset_incoming_trained_models()
            return

        self.aggregation_durations[round] = time.time() - start_time

        # 3.1. Aggregate these models
        self.logger.info("Aggregator %s will average the models of round %d",
                         self.peer_manager.get_my_short_id(), round)
        avg_model = self.model_manager.average_trained_models()

        # 3.2. Remove these models from the model manager (they are not needed anymore)
        self.model_manager.reset_incoming_trained_models()

        # 3. Determine the aggregators of the next sample that are available
        participants = await self.determine_available_participants_for_round(round + 1)
        participants_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in participants]
        self.logger.info("Participant %s determined %d available participants for round %d: %s",
                         self.peer_manager.get_my_short_id(), len(participants_ids), round + 1, participants_ids)

        # 3.3. Distribute the average model to the available participants in the sample.
        await self.send_aggregated_model_to_participants(participants, avg_model, self.sample_index_estimate)

        # 4. As last step, send the model to yourself. We want this task to be finished before doing so.
        if self.my_id in participants:
            model_cpy = copy.deepcopy(avg_model)
            asyncio.get_event_loop().call_soon(self.received_aggregated_model, self.my_peer, self.sample_index_estimate, model_cpy)

        return avg_model

    async def send_aggregated_model_to_participants(self, participants: List[bytes], model: nn.Module, sample_index: int) -> None:
        if not self.is_active:
            self.logger.warning("Participant %s not sending aggregated model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        self.logger.info("Participant %s sending aggregated model of round %d to participants",
                         self.peer_manager.get_my_short_id(), sample_index)

        # For load balancing purposes, shuffle this list
        random.shuffle(participants)

        population_view = copy.deepcopy(self.peer_manager.last_active)
        for peer_pk in participants:
            if peer_pk == self.my_id:
                continue

            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue
            await self.eva_send_model(sample_index, model, "aggregated_model", population_view, peer)

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()
        self.update_population_view_history()

    async def send_trained_model_to_aggregators(self, aggregators: List[bytes], sample_index: int) -> None:
        """
        Send the current model to the aggregators in a particular sample.
        """
        if not self.is_active:
            self.logger.warning("Participant %s not sending trained model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        aggregator_ids = [self.peer_manager.get_short_id(aggregator) for aggregator in aggregators]
        self.logger.info("Participant %s sending trained model of round %d to %d aggregators in sample %d: %s",
                         self.peer_manager.get_my_short_id(), sample_index - 1, len(aggregators), sample_index, aggregator_ids)
        population_view = copy.deepcopy(self.peer_manager.last_active)

        # For load balancing purposes, shuffle this list
        random.shuffle(aggregators)

        for aggregator in aggregators:
            if aggregator == self.my_id:
                continue

            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue
            await self.eva_send_model(sample_index, self.model_manager.model, "trained_model", population_view, peer)

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

    def on_eva_send_done(self, future: Future, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float):
        if future.exception():
            peer_id = self.peer_manager.get_short_id(peer.public_key.key_to_bin())
            self.logger.warning("Transfer to participant %s failed, scheduling it again (Exception: %s)",
                                peer_id, future.exception())
            # The transfer failed - try it again after some delay
            ensure_future(asyncio.sleep(self.model_send_delay)).add_done_callback(
                lambda _: self.schedule_eva_send_model(peer, serialized_response, binary_data, start_time))
        else:
            # The transfer seems to be completed - record the transfer time
            self.transfer_times.append(time.time() - start_time)

    def schedule_eva_send_model(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float):
        # Schedule the transfer
        future = self.eva.send_binary(peer, serialized_response, binary_data)
        ensure_future(future).add_done_callback(
            lambda f: self.on_eva_send_done(f, peer, serialized_response, binary_data, start_time))

    async def eva_send_model(self, round, model, type, population_view, peer):
        start_time = time.time()
        serialized_model = serialize_model(model)
        serialized_population_view = pickle.dumps(population_view)
        binary_data = serialized_model + serialized_population_view
        response = {"round": round, "type": type, "model_data_len": len(serialized_model)}
        serialized_response = json.dumps(response).encode()
        self.schedule_eva_send_model(peer, serialized_response, binary_data, start_time)

    def cancel_current_aggregation_task(self):
        if self.ongoing_aggregation_task_name and self.is_pending_task_active(self.ongoing_aggregation_task_name):
            self.logger.info("Participant %s interrupting aggregation task %s",
                             self.peer_manager.get_my_short_id(), self.ongoing_training_task_name)
            self.cancel_pending_task(self.ongoing_aggregation_task_name)
            self.model_manager.reset_incoming_trained_models()
            self.ongoing_aggregation_task_name = None

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
            self.logger.warning("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())
        serialized_model = result.data[:json_data["model_data_len"]]
        serialized_population_view = result.data[json_data["model_data_len"]:]
        received_population_view = pickle.loads(serialized_population_view)
        self.peer_manager.merge_population_views(received_population_view)
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(),
                                               max(json_data["round"], self.sample_index_estimate))
        self.update_population_view_history()
        incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])

        if json_data["type"] == "trained_model":
            self.received_trained_model(result.peer, json_data["round"], incoming_model)
        elif json_data["type"] == "aggregated_model":
            self.received_aggregated_model(result.peer, json_data["round"], incoming_model)

    def received_trained_model(self, peer: Peer, model_round: int, model: nn.Module) -> None:
        if self.shutting_down:
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received trained model for round %d from participant %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        if model_round > self.sample_index_estimate:
            self.logger.info("Participant %s received trained model for round %d for the first time - "
                             "starting to aggregate", self.peer_manager.get_my_short_id(), model_round)
            self.cancel_current_aggregation_task()  # Interrupt current aggregation work
            self.sample_index_estimate = model_round
            self.model_manager.process_incoming_trained_model(peer_pk, model)
            self.aggregate_in_round(model_round)
        elif model_round == self.sample_index_estimate:
            self.model_manager.process_incoming_trained_model(peer_pk, model)
            if self.model_manager.has_enough_trained_models() and self.aggregation_future and not self.aggregation_future.done():
                self.logger.info("Aggregator %s received sufficient trained models (%d) of round %d",
                                 self.peer_manager.get_my_short_id(), len(self.model_manager.incoming_trained_models), model_round)
                self.aggregation_future.set_result(None)

    def received_aggregated_model(self, peer: Peer, model_round: int, model: nn.Module) -> None:
        if self.shutting_down:
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received aggregated model of round %d from aggregator %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        if model_round > self.sample_index_estimate:
            self.cancel_current_training_task()
            self.sample_index_estimate = model_round
        if model_round == self.sample_index_estimate and not self.ongoing_training_task_name:
            self.model_manager.model = model
            self.train_in_round(model_round)

    async def on_send_complete(self, result: TransferResult):
        peer_id = self.peer_manager.get_short_id(result.peer.public_key.key_to_bin())
        my_peer_id = self.peer_manager.get_my_short_id()
        self.logger.info(f'Outgoing transfer {my_peer_id} -> {peer_id} has completed: {result.info.decode()}')
        self.peer_manager.update_peer_activity(result.peer.public_key.key_to_bin(), self.sample_index_estimate)

    async def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')

    async def unload(self):
        self.shutting_down = True
        await self.request_cache.shutdown()
        await super().unload()
