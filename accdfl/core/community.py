import asyncio
import json
import os
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from typing import Optional, Dict, Set

from accdfl.core import TransmissionMethod
from accdfl.core.model import serialize_model, unserialize_model, create_model
from accdfl.core.dataset import TrainDataset
from accdfl.core.model_manager import ModelManager
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.peer_manager import PeerManager
from accdfl.core.sample_manager import SampleManager
from accdfl.util.eva_protocol import EVAProtocolMixin, TransferResult

from ipv8.community import Community
from ipv8.types import Peer


class DFLCommunity(EVAProtocolMixin, Community):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_id = self.my_peer.public_key.key_to_bin()
        self.is_active = False
        self.model_send_delay = None
        self.round_complete_callback = None
        self.parameters = None
        self.participating_in_rounds: Set[int] = set()
        self.aggregating_in_rounds: Set[int] = set()
        self.sample_size = None
        self.did_setup = False
        self.shutting_down = False

        self.peer_manager: PeerManager = PeerManager(self.my_id)
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup
        self.model_manager: Optional[ModelManager] = None    # Initialized when the process is setup

        self.aggregation_deferred = Future()

        # Model exchange parameters
        self.data_dir = None
        self.transmission_method = TransmissionMethod.EVA
        self.eva_max_retry_attempts = 20

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

    def setup(self, parameters: Dict, data_dir: str, transmission_method: TransmissionMethod = TransmissionMethod.EVA):
        assert len(parameters["participants"]) * parameters["local_classes"] == sum(parameters["nodes_per_class"])

        self.parameters = parameters
        self.data_dir = data_dir
        self.sample_size = parameters["sample_size"]
        self.logger.info("Setting up experiment with %d participants and sample size %d (I am participant %s)" %
                         (len(parameters["participants"]), self.sample_size, self.peer_manager.get_my_short_id()))

        for participant in parameters["participants"]:
            self.peer_manager.add_peer(unhexlify(participant))
        self.sample_manager = SampleManager(self.peer_manager, self.sample_size, parameters["num_aggregators"])

        # Initialize the model
        participant_index = parameters["participants"].index(hexlify(self.my_id).decode())
        model = create_model(parameters["dataset"], parameters["model"])
        dataset = TrainDataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters, participant_index)
        optimizer = SGDOptimizer(model, parameters["learning_rate"], parameters["momentum"])
        self.model_manager = ModelManager(model, dataset, optimizer, parameters)

        # Setup the model transmission
        self.transmission_method = transmission_method
        if self.transmission_method == TransmissionMethod.EVA:
            self.logger.info("Setting up EVA protocol")
            self.eva_init(window_size_in_blocks=32, retransmit_attempt_count=10, retransmit_interval_in_sec=1,
                          timeout_interval_in_sec=10)
            self.eva_register_receive_callback(self.on_receive)
            self.eva_register_send_complete_callback(self.on_send_complete)
            self.eva_register_error_callback(self.on_error)

        self.did_setup = True

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    async def participate_in_round(self, round):
        """
        Participate in a round.
        """
        self.logger.error("START %d by %s", round, self.peer_manager.get_my_short_id())
        if round < 1:
            raise RuntimeError("Round number %d invalid!" % round)

        if round in self.participating_in_rounds:
            raise RuntimeError("Want to start a round (%d) that is already ongoing" % round)

        self.participating_in_rounds.add(round)
        sample_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in self.sample_manager.get_sample_for_round(round)]
        aggregator_ids = [self.peer_manager.get_short_id(peer_id) for peer_id in self.sample_manager.get_aggregators_for_round(round + 1)]
        self.logger.info("Participant %s starts participating in round %d (sample: %s, aggregators for next round: %s)",
                         self.peer_manager.get_my_short_id(), round, sample_ids, aggregator_ids)

        # 1. Train the model
        self.model_manager.train()

        # 2. Send the model to the aggregators of the next round
        await self.send_trained_model_to_aggregators(round)

        # 3. Complete the round
        self.logger.info("Participant %s completed round %d", self.peer_manager.get_my_short_id(), round)
        self.participating_in_rounds.remove(round)
        if self.round_complete_callback:
            self.round_complete_callback(round)

    async def aggregate_in_round(self, round: int):
        self.logger.info("Aggregator %s starts aggregating in round %d", self.peer_manager.get_my_short_id(), round)
        self.aggregating_in_rounds.add(round)

        if not self.model_manager.has_enough_trained_models_of_round(round):
            self.logger.info("Aggregator %s start to wait for trained models of round %d",
                             self.peer_manager.get_my_short_id(), round)
            await asyncio.wait_for(self.aggregation_deferred, timeout=5.0)
            self.aggregation_deferred = Future()

        # 3.1. Aggregate these models
        self.logger.info("Aggregator %s will average the models of round %d",
                         self.peer_manager.get_my_short_id(), round)
        avg_model = self.model_manager.average_trained_models_of_round(round)
        self.model_manager.adopt_model(avg_model)

        # 3.2. Remove these models from the model manager (they are not needed anymore)
        self.model_manager.remove_trained_models_of_round(round)

        # 3.3. Distribute the average model to the non-aggregator nodes in the sample.
        this_round_participants = self.sample_manager.get_sample_for_round(round + 1)
        this_round_aggregators = self.sample_manager.get_aggregators_for_round(round + 1)
        for peer_pk in [p for p in this_round_participants if p not in this_round_aggregators]:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                continue
            await self.eva_send_model(round, avg_model, "aggregated_model", peer)

        self.logger.info("Aggregator %s completed aggregation in round %d", self.peer_manager.get_my_short_id(), round)
        self.aggregating_in_rounds.remove(round)

        # 5. If we were an aggregator in this round, we are a participant in the next once.
        # Since we won't receive a trigger message, start the next round.
        if self.sample_manager.is_aggregator_in_round(self.my_id, round + 1) and (
                round + 1) not in self.participating_in_rounds:
            self.register_task("round_%d" % (round + 1), self.participate_in_round, round + 1)

    async def send_trained_model_to_aggregators(self, round: int) -> None:
        """
        Send the current model to the aggregators of the sample associated with the next round.
        """
        aggregators = self.sample_manager.get_aggregators_for_round(round + 1)
        aggregator_ids = [self.peer_manager.get_short_id(aggregator) for aggregator in aggregators]
        self.logger.info("Participant %s sending trained model of round %d to %d aggregators: %s",
                         self.peer_manager.get_my_short_id(), round, len(aggregators), aggregator_ids)
        for aggregator in aggregators:
            if aggregator == self.my_id:
                serialized_model = serialize_model(self.model_manager.model)
                ensure_future(self.received_trained_model(self.my_peer, round, serialized_model))
                continue

            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue
            await self.eva_send_model(round, self.model_manager.model, "trained_model", peer)

    async def eva_send_model(self, round, model, type, peer):
        response = {"round": round, "type": type}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            self.logger.info("Participant %s sending round %d model to participant %s (attempt %d)",
                             self.peer_manager.get_my_short_id(), round,
                             self.peer_manager.get_short_id(peer.public_key.key_to_bin()), attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once?
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(model))
                self.logger.info("Participant %s successfully sent %s of round %s to participant %s",
                                 self.peer_manager.get_my_short_id(), type, round,
                                 self.peer_manager.get_short_id(peer.public_key.key_to_bin()))
                break
            except Exception:
                self.logger.exception("Exception when sending aggregated model to peer %s", peer)
            attempt += 1

    def on_receive(self, result: TransferResult):
        assert result.peer.public_key.key_to_bin() in self.peer_manager.peers

        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()
        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())
        if json_data["type"] == "trained_model":
            ensure_future(self.received_trained_model(result.peer, json_data["round"], result.data))
        elif json_data["type"] == "aggregated_model":
            self.received_aggregated_model(result.peer, json_data["round"], result.data)

    async def received_trained_model(self, peer: Peer, model_round: int, serialized_model: bytes) -> None:
        if self.shutting_down:
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received trained model for round %d from participant %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        # Check if the peer that sent us the trained model is indeed a participant in the round.
        if not self.sample_manager.is_participant_in_round(peer_pk, model_round):
            self.logger.warning("Participant %s is not a participant in round %d", peer_id, model_round)
            return

        # Check if we are an aggregator in the next round
        if not self.sample_manager.is_aggregator_in_round(self.my_id, model_round + 1):
            self.logger.warning("Participant %s is not an aggregator in round %d",
                                self.peer_manager.get_my_short_id(), model_round + 1)
            return

        # Start the aggregation logic if we haven't done yet.
        task_name = "aggregate_%d" % model_round
        if model_round not in self.aggregating_in_rounds and not self.is_pending_task_active(task_name):
            self.register_task(task_name, self.aggregate_in_round, model_round)

        # Process the model
        incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
        self.model_manager.process_incoming_trained_model(peer_pk, model_round, incoming_model)

        # Do we have enough models to start aggregating the models and send them to the other peers in the sample?
        # TODO integrate the success factor
        if self.model_manager.has_enough_trained_models_of_round(model_round):
            self.logger.info("Aggregator %s received sufficient trained models of round %d",
                             self.peer_manager.get_my_short_id(), model_round)
            self.aggregation_deferred.set_result(None)

    def received_aggregated_model(self, peer: Peer, model_round: int, serialized_model: bytes) -> None:
        if self.shutting_down:
            return

        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Participant %s received aggregated model of round %d from aggregator %s",
                         self.peer_manager.get_my_short_id(), model_round, peer_id)

        # Check if the peer that sent us the aggregated model is indeed an aggregator in this round.
        if not self.sample_manager.is_aggregator_in_round(peer_pk, model_round + 1):
            self.logger.warning("Participant %s is not an aggregator in round %d", peer_id, model_round + 1)
            return

        # Check if we are a participant in this round.
        if not self.sample_manager.is_participant_in_round(self.my_id, model_round + 1):
            self.logger.warning("Participant %s is not a participant in round %d",
                                self.peer_manager.get_my_short_id(), model_round)
            return

        # If this is the first time we receive an aggregated model for this round, adopt the model and start
        # participating in the next round.
        if (model_round + 1) not in self.participating_in_rounds and not self.is_pending_task_active("round_%d" % (model_round + 1)):
            # TODO we are not waiting on all models from other aggregators. We might want to do this in the future to make the system more robust.
            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            self.model_manager.adopt_model(incoming_model)
            self.register_task("round_%d" % (model_round + 1), self.participate_in_round, model_round + 1)

    def on_send_complete(self, result: TransferResult):
        peer_id = self.peer_manager.get_short_id(result.peer.public_key.key_to_bin())
        self.logger.info(f'Outgoing transfer to participant {peer_id} has completed: {result.info.decode()}')

    def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')

    async def unload(self):
        self.shutting_down = True
        await super().unload()