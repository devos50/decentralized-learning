import asyncio
import copy
import json
import pickle
import random
import time
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from typing import Optional, Dict, Set, List, Callable

from torch import nn

from accdfl.core import TransmissionMethod, NodeMembershipChange
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
        self.is_active = False
        self.model_send_delay = 1.0
        self.round_complete_callback: Optional[Callable] = None
        self.aggregate_complete_callback: Optional[Callable] = None
        self.parameters = None
        self.participating_in_rounds: Set[int] = set()
        self.aggregating_in_rounds: Set[int] = set()
        self.aggregation_durations: Dict[int, float] = {}
        self.last_round_completed: int = 0
        self.last_aggregate_round_completed: int = 0
        self.sample_size = None
        self.did_setup = False
        self.shutting_down = False
        self.train_in_subprocess = True
        self.fixed_aggregator = None

        self.peer_manager: PeerManager = PeerManager(self.my_id, -1)
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup
        self.model_manager: Optional[ModelManager] = None    # Initialized when the process is setup

        self.aggregation_futures: Dict[int, Future] = {}

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

    def start(self):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"

        self.is_active = True

        # Start the process
        if self.sample_manager.is_participant_in_round(self.my_id, 1):
            self.register_task("round_1", self.participate_in_round, 1)
        else:
            self.logger.info("Participant %s won't participate in round 1", self.peer_manager.get_my_short_id())

    def setup(self, parameters: Dict, data_dir: str, transmission_method: TransmissionMethod = TransmissionMethod.EVA,
              aggregator: Optional[bytes] = None):
        assert parameters["target_participants"] * parameters["local_classes"] == sum(parameters["nodes_per_class"])

        self.parameters = parameters
        self.data_dir = data_dir
        self.fixed_aggregator = aggregator
        self.sample_size = parameters["sample_size"]
        self.logger.info("Setting up experiment with %d initial participants and sample size %d (I am participant %s)" %
                         (len(parameters["participants"]), self.sample_size, self.peer_manager.get_my_short_id()))

        self.peer_manager.inactivity_threshold = parameters["inactivity_threshold"]
        for participant in parameters["participants"]:
            self.peer_manager.add_peer(unhexlify(participant))
        self.sample_manager = SampleManager(self.peer_manager, self.sample_size, parameters["num_aggregators"])

        # Initialize the model
        participant_index = parameters["participants"].index(hexlify(self.my_id).decode())
        model = create_model(parameters["dataset"], parameters["model"])
        self.model_manager = ModelManager(model, parameters, participant_index)

        # Setup the model transmission
        self.transmission_method = transmission_method
        if self.transmission_method == TransmissionMethod.EVA:
            self.logger.info("Setting up EVA protocol")
            self.eva.settings.window_size = 64
            self.eva.settings.retransmit_attempt_count = 10
            self.eva.settings.retransmit_interval_in_sec = 1
            self.eva.settings.timeout_interval_in_sec = 10

        self.did_setup = True

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    def go_offline(self, round: int, graceful: bool = True) -> None:
        self.is_active = False
        self.cancel_all_pending_tasks()

        self.logger.info("Participant %s will go offline in round %d", self.peer_manager.get_my_short_id(), round)

        info = self.peer_manager.last_active[self.my_id]
        self.peer_manager.last_active[self.my_id] = (info[0], (round, NodeMembershipChange.LEAVE))
        if graceful:
            self.advertise_membership(round, NodeMembershipChange.LEAVE)

    def advertise_membership(self, round: int, change: NodeMembershipChange):
        """
        Advertise your (new) membership to the peers in a particular round.
        """
        # Note that we have to send this to the sample WITHOUT considering the newly joined node!
        participants_in_sample = self.sample_manager.get_sample_for_round(round, exclude_peer=self.my_id)
        for participant in participants_in_sample:
            if participant == self.my_id:
                continue

            self.logger.info("Participant %s advertising its membership change to participant %s (part of round %d)",
                              self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(participant), round)
            peer = self.get_peer_by_pk(participant)
            global_time = self.claim_global_time()
            auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
            payload = AdvertiseMembership(round - 1, change.value)
            dist = GlobalTimeDistributionPayload(global_time)
            packet = self._ez_pack(self._prefix, AdvertiseMembership.msg_id, [auth, dist, payload])
            self.endpoint.send(peer.address, packet)

    @lazy_wrapper(GlobalTimeDistributionPayload, AdvertiseMembership)
    def on_membership_advertisement(self, peer, dist, payload):
        """
        We received a membership advertisement from a new peer.
        """
        # TODO we assume that the peer is allowed to participate
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        self.logger.info("Participant %s updating membership of participant %s",
                         self.peer_manager.get_my_short_id(), peer_id)

        change: NodeMembershipChange = NodeMembershipChange(payload.change)
        if change == NodeMembershipChange.JOIN:
            # Do not apply this immediately since we do not want the newly joined node to be part of the next sample just yet.
            self.peer_manager.last_active_pending[peer_pk] = (payload.round, (payload.round, NodeMembershipChange.JOIN))
        else:
            self.peer_manager.last_active[peer_pk] = (payload.round, (payload.round, NodeMembershipChange.LEAVE))

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

        cache = PingRequestCache(self, ping_all_id, peer_pk, round, self.parameters["ping_timeout"])
        self.request_cache.add(cache)
        self.send_ping(peer, round, cache.number)
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
        peer_short_id = self.peer_manager.get_short_id(peer.public_key.key_to_bin())
        if not self.request_cache.has("ping-%s" % peer_short_id, payload.identifier):
            self.logger.warning("ping cache with id %s not found", payload.identifier)
            return

        # Update the population view with this new information
        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(), payload.round)

        cache = self.request_cache.pop("ping-%s" % peer_short_id, payload.identifier)
        cache.future.set_result((peer.public_key.key_to_bin(), payload.round, True))

    async def participate_in_round(self, round):
        """
        Participate in a round.
        """
        if round < 1:
            raise RuntimeError("Round number %d invalid!" % round)

        if round in self.participating_in_rounds:
            raise RuntimeError("Want to start a round (%d) that is already ongoing" % round)

        self.participating_in_rounds.add(round)
        sample_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in self.sample_manager.get_sample_for_round(round)]
        self.logger.info("Participant %s starts participating in round %d (sample: %s)",
                         self.peer_manager.get_my_short_id(), round, sample_ids)

        # 1. Train the model
        await self.model_manager.train(self.train_in_subprocess)

        # 2. Determine the aggregators of the next sample that are available
        aggregators = await self.determine_available_aggregators_for_round(round + 1)
        aggregator_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in aggregators]
        self.logger.info("Participant %s determined %d available aggregators for round %d: %s",
                         self.peer_manager.get_my_short_id(), len(aggregator_ids), round + 1, aggregator_ids)

        # 2. Send the model to these available aggregators
        await self.send_trained_model_to_aggregators(aggregators, round)

        # 4. Complete the round
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round)
        self.participating_in_rounds.remove(round)
        self.last_round_completed = max(self.last_round_completed, round)
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(round))

    async def aggregate_in_round(self, round: int):
        self.logger.info("Aggregator %s starts aggregating in round %d", self.peer_manager.get_my_short_id(), round)
        self.aggregating_in_rounds.add(round)
        start_time = time.time()

        if not self.model_manager.has_enough_trained_models_of_round(round):
            self.logger.info("Aggregator %s starts to wait for trained models of round %d",
                             self.peer_manager.get_my_short_id(), round)
            self.aggregation_futures[round] = Future()
            received_sufficient_models = False
            try:
                await asyncio.wait_for(self.aggregation_futures[round], timeout=self.parameters["aggregation_timeout"])
                received_sufficient_models = True
            except asyncio.exceptions.TimeoutError:
                self.logger.warning("Aggregator %s triggered timeout while waiting for models of round %d",
                                    self.peer_manager.get_my_short_id(), round)
            except StopAggregationException:
                self.logger.warning("Aggregator %s triggered StopAggregationException while waiting for models of "
                                    "round %d", self.peer_manager.get_my_short_id(), round)
            self.aggregation_futures.pop(round, None)
        else:
            received_sufficient_models = True

        if not received_sufficient_models:
            self.model_manager.remove_trained_models_of_round(round)
            self.aggregating_in_rounds.remove(round)
            self.last_aggregate_round_completed = max(self.last_aggregate_round_completed, round)
            return

        self.aggregation_durations[round] = time.time() - start_time

        # 3.1. Aggregate these models
        self.logger.info("Aggregator %s will average the models of round %d",
                         self.peer_manager.get_my_short_id(), round)
        avg_model = self.model_manager.average_trained_models_of_round(round)

        # 3.2. Remove these models from the model manager (they are not needed anymore)
        self.model_manager.remove_trained_models_of_round(round)

        # 3.3. Distribute the average model to the available participants in the sample.
        await self.send_aggregated_model_to_participants(avg_model, round)

        self.logger.info("Aggregator %s completed aggregation in round %d", self.peer_manager.get_my_short_id(), round)
        self.aggregating_in_rounds.remove(round)
        self.last_aggregate_round_completed = max(self.last_aggregate_round_completed, round)

        # 4. Invoke the callback
        if self.aggregate_complete_callback:
            ensure_future(self.aggregate_complete_callback(round, avg_model))

    async def send_aggregated_model_to_participants(self, model: nn.Module, model_round: int) -> None:
        if not self.is_active:
            self.logger.warning("Participant %s not sending aggregated model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        self.logger.info("Participant %s sending aggregated model of round %d to participants",
                         self.peer_manager.get_my_short_id(), model_round)

        # Determine the available participants for the next round
        participants = await self.determine_available_participants_for_round(model_round + 1)
        participants_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in participants]
        self.logger.info("Participant %s determined %d available participants for round %d: %s",
                         self.peer_manager.get_my_short_id(), len(participants_ids), model_round + 1, participants_ids)

        # For load balancing purposes, shuffle this list
        random.shuffle(participants)

        population_view = copy.deepcopy(self.peer_manager.last_active)
        for peer_pk in participants:
            if peer_pk == self.my_id:
                model_cpy = copy.deepcopy(model)
                self.received_aggregated_model(self.my_peer, model_round, model_cpy)
                continue

            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue
            await self.eva_send_model(model_round, model, "aggregated_model", population_view, peer)

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

    async def send_trained_model_to_aggregators(self, aggregators: List[bytes], round: int) -> None:
        """
        Send the current model to the aggregators of the sample associated with the next round.
        """
        if not self.is_active:
            self.logger.warning("Participant %s not sending trained model due to offline status",
                                self.peer_manager.get_my_short_id())
            return

        aggregator_ids = [self.peer_manager.get_short_id(aggregator) for aggregator in aggregators]
        self.logger.info("Participant %s sending trained model of round %d to %d aggregators: %s",
                         self.peer_manager.get_my_short_id(), round, len(aggregators), aggregator_ids)
        population_view = copy.deepcopy(self.peer_manager.last_active)

        # For load balancing purposes, shuffle this list
        random.shuffle(aggregators)

        for aggregator in aggregators:
            if aggregator == self.my_id:
                self.received_trained_model(self.my_peer, round, self.model_manager.model)
                continue

            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue
            await self.eva_send_model(round, self.model_manager.model, "trained_model", population_view, peer)

        # Flush pending changes to the local view
        self.peer_manager.flush_last_active_pending()

    def on_eva_send_done(self, future: Future, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float):
        if future.exception():
            peer_id = self.peer_manager.get_short_id(peer.public_key.key_to_bin())
            self.logger.warning("Transfer to participant %s failed, scheduling it again (Exception: %s)",
                                peer_id, future.exception())
            # The transfer failed - try it again after some delay
            asyncio.sleep(self.model_send_delay).add_done_callback(
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
        self.peer_manager.update_last_active(received_population_view)
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
        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(), model_round)

        self.logger.info("Participant %s received trained model for round %d from participant %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        # TODO Check if the peer that sent us the trained model is indeed a participant in the round.
        # This is more difficult with dynamic populations since we cannot reliably check this based on received information.

        # Start the aggregation logic if we haven't done yet.
        task_name = "aggregate_%d" % model_round
        if model_round not in self.aggregating_in_rounds and not self.is_pending_task_active(task_name) and self.last_aggregate_round_completed < model_round:
            self.register_task(task_name, self.aggregate_in_round, model_round)

        # Process the model
        self.model_manager.process_incoming_trained_model(peer_pk, model_round, model)

        # Do we have enough models to start aggregating the models and send them to the other peers in the sample?
        # TODO integrate the success factor
        if self.model_manager.has_enough_trained_models_of_round(model_round):
            # It could be that the register_task call above is slower than this logic.
            if model_round in self.aggregation_futures:
                self.logger.info("Aggregator %s received sufficient trained models of round %d",
                                 self.peer_manager.get_my_short_id(), model_round)
                if not self.aggregation_futures[model_round].done():
                    self.aggregation_futures[model_round].set_result(None)

    def received_aggregated_model(self, peer: Peer, model_round: int, model: nn.Module) -> None:
        if self.shutting_down:
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        self.peer_manager.update_peer_activity(peer.public_key.key_to_bin(), model_round)

        self.logger.info("Participant %s received aggregated model of round %d from aggregator %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        # If we are aggregating in this particular model round and still waiting for models, stop doing so.
        if model_round in self.aggregating_in_rounds and model_round in self.aggregation_futures and \
                not self.aggregation_futures[model_round].done():
            self.logger.warning("Received aggregated model for round %d from participant %s while were still waiting "
                                "for trained models in that round! Stopping to wait", model_round, peer_id)
            self.aggregation_futures[model_round].set_exception(StopAggregationException())

            # Help to spread this received aggregated model
            ensure_future(self.send_aggregated_model_to_participants(model, model_round))

        # If this is the first time we receive an aggregated model for this round, adopt the model and start
        # participating in the next round.
        next_round = (model_round + 1)
        task_name = "round_%d" % next_round
        if next_round not in self.participating_in_rounds and not self.is_pending_task_active(task_name) and self.last_round_completed < next_round:
            # TODO we are not waiting on all models from other aggregators. We might want to do this in the future to make the system more robust.
            self.model_manager.model = model
            self.register_task(task_name, self.participate_in_round, next_round)

    async def on_send_complete(self, result: TransferResult):
        peer_id = self.peer_manager.get_short_id(result.peer.public_key.key_to_bin())
        self.logger.info(f'Outgoing transfer to participant {peer_id} has completed: {result.info.decode()}')

    async def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')

    async def unload(self):
        self.shutting_down = True
        await self.request_cache.shutdown()
        await super().unload()
