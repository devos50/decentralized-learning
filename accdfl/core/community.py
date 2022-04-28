import json
import os
from asyncio import Future, ensure_future
from binascii import unhexlify, hexlify
from typing import Optional, Dict, List

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
        self.is_participating_in_round = False
        self.model_send_delay = None
        self.round_complete_callback = None
        self.parameters = None
        self.round = 1  # TODO Not sure if this should be a class variable - in practice a node can be active in multiple nodes at the same time
        self.sample_size = None
        self.did_setup = False

        self.peer_manager: PeerManager = PeerManager(self.my_id)
        self.sample_manager: Optional[SampleManager] = None  # Initialized when the process is setup
        self.model_manager: Optional[ModelManager] = None    # Initialized when the process is setup

        self.round_deferred = Future()
        self.incoming_aggregated_models: Dict[int, List] = {}
        self.compute_accuracy_deferred = None

        # Model exchange parameters
        self.data_dir = None
        self.transmission_method = TransmissionMethod.EVA
        self.eva_max_retry_attempts = 20

        self.logger.info("The DFL community started with public key: %s",
                         hexlify(self.my_peer.public_key.key_to_bin()).decode())

    def start(self):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"

        self.is_active = True

        # Start the process
        if self.sample_manager.is_participant_in_round(self.my_id, self.round):
            ensure_future(self.participate_in_round())
        else:
            self.logger.info("Participant %s won't participate in round %d", self.peer_manager.get_my_short_id(), self.round)

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

    async def participate_in_round(self):
        """
        Complete a round of training and model aggregation.
        """
        self.is_participating_in_round = True
        self.logger.info("Participant %s starts participating in round %d", self.peer_manager.get_my_short_id(),
                         self.round)

        # Train the model
        self.model_manager.train()

        # Send the trained model to the aggregators of the next sample
        await self.send_model_to_aggregators()

    async def send_model_to_aggregators(self):
        aggregators = self.sample_manager.get_aggregators_for_round(self.round + 1)
        self.logger.info("Sending updated model of round %d to %d aggregators" % (self.round, len(aggregators)))
        for aggregator in aggregators:
            peer = self.get_peer_by_pk(aggregator)
            if not peer:
                self.logger.warning("Could not find aggregator peer with public key %s", hexlify(aggregator).decode())
                continue
            await self.eva_send_model(self.round, self.model_manager.model, "trained_model", peer)

    async def eva_send_model(self, round, model, type, peer):
        response = {"round": round, "type": type}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            self.logger.info("Participant %d sending round %d model to participant %d (attempt %d)",
                             self.peer_manager.get_my_short_id(), round,
                             self.peer_manager.get_short_id(peer.public_key.key_to_bin()), attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once?
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(model))
                self.logger.info("Participant %d successfully sent model of round %d to participant %s",
                                 self.peer_manager.get_my_short_id(), round,
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
            self.received_trained_model(result.peer, json_data["round"], result.data)
        elif json_data["type"] == "aggregated_model":
            self.received_aggregated_model(result.peer, json_data["round"], result.data)

    async def received_trained_model(self, peer: Peer, model_round: int, serialized_model: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Received local model for round %d from participant %s", model_round, peer_id)

        # Check if the peer that sent us the trained model is indeed a participant in the round.
        if not self.sample_manager.is_participant_in_round(peer_pk, model_round):
            self.logger.warning("Participant %s is not a participant in round %d", peer_id, model_round)
            return

        # Check if we are an aggregator in the next round
        if not self.sample_manager.is_aggregator_in_round(peer_pk, model_round + 1):
            self.logger.warning("We are not an aggregator in round %d", model_round)
            return

        # Process the model
        incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
        self.model_manager.process_incoming_trained_model(peer_pk, model_round, incoming_model)

        # Do we have enough models to start aggregating the models and send them to the other peers in the sample?
        # TODO integrate the success factor
        if self.model_manager.has_enough_trained_models_of_round(model_round):
            # Aggregate these models and distribute them to the non-aggregator nodes in the sample.
            avg_model = self.model_manager.average_trained_models_of_round(model_round)
            self.model_manager.adopt_model(avg_model)

            next_round = model_round + 1
            next_round_participants = self.sample_manager.get_sample_for_round(next_round)
            next_round_aggregators = self.sample_manager.get_aggregators_for_round(next_round)
            for peer_pk in [p for p in next_round_participants if p not in next_round_aggregators]:
                peer = self.get_peer_by_pk(peer_pk)
                if not peer:
                    self.logger.warning("Could not find peer with public key %s", hexlify(peer_pk).decode())
                    continue
                await self.eva_send_model(self.round, avg_model, "aggregated_model", peer)

            self.round = next_round
            ensure_future(self.participate_in_round())

    def received_aggregated_model(self, peer: Peer, model_round: int, serialized_model: bytes) -> None:
        peer_pk = peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)

        self.logger.info("Received aggregated model for round %d from aggregator %s", model_round, peer_id)

        # Check if the peer that sent us the trained model is indeed an aggregator in this round.
        if not self.sample_manager.is_aggregator_in_round(peer_pk, model_round):
            self.logger.warning("Participant %s is not an aggregator in round %d", peer_id, model_round)
            return

        # Check if we are a participant in this round.
        if not self.sample_manager.is_participant_in_round(peer_pk, model_round):
            self.logger.warning("We are not a participant in round %d", model_round)
            return

        # If this is the first time we receive an aggregated model for this round, adopt the model and start training.
        if self.round < model_round:
            # TODO we are not waiting on all models from other aggregators. We might want to do this in the future to make the system more robust.
            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            self.model_manager.adopt_model(incoming_model)
            self.round = model_round
            ensure_future(self.participate_in_round())

    def on_send_complete(self, result: TransferResult):
        peer_id = self.peer_manager.get_short_id(result.peer.public_key.key_to_bin())
        self.logger.info(f'Outgoing transfer to participant {peer_id} has completed: {result.info.decode()}')

    def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')
