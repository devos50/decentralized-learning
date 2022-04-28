import copy
import itertools
import json
import os
import random
from asyncio import Future, sleep, ensure_future
from binascii import unhexlify, hexlify
from enum import Enum
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from accdfl.core.model import serialize_model, unserialize_model, create_model
from accdfl.core.dataset import Dataset
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.util.eva_protocol import EVAProtocolMixin, TransferResult
from ipv8.community import Community


class TransmissionMethod(Enum):
    EVA = 0


class DFLCommunity(EVAProtocolMixin, Community):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_active = False
        self.is_participating_in_round = False
        self.model_send_delay = None
        self.round_complete_callback = None
        self.parameters = None
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.round = 1
        self.epoch = 1
        self.sample_size = None
        self.participants: Optional[List[str]] = None
        self.round_deferred = Future()
        self.incoming_local_models: Dict[int, List] = {}
        self.incoming_aggregated_models: Dict[int, List] = {}
        self.is_computing_accuracy = False
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
        self.is_active = True

        # Start the process
        if self.is_participant_for_round(self.round):
            ensure_future(self.participate_in_round())
        else:
            self.logger.info("Participant %d won't participate in round %d", self.get_my_participant_index(), self.round)

    def get_participant_index(self, public_key: bytes) -> int:
        if not self.participants:
            return -1
        return self.participants.index(hexlify(public_key).decode())

    def get_my_participant_index(self):
        return self.get_participant_index(self.my_peer.public_key.key_to_bin())

    def get_participants_for_round(self, round: int) -> List[int]:
        rand = random.Random(round)
        participant_indices = list(range(len(self.participants)))
        return sorted(rand.sample(participant_indices, self.sample_size))

    def is_participant_for_round(self, round: int) -> bool:
        return self.get_my_participant_index() in self.get_participants_for_round(round)

    def get_round_representative(self, round: int) -> int:
        rand = random.Random(round)
        return rand.choice(self.get_participants_for_round(round))

    def is_round_representative(self, round: int) -> bool:
        return self.get_my_participant_index() == self.get_round_representative(round)

    def setup(self, parameters: Dict, data_dir: str, transmission_method: TransmissionMethod = TransmissionMethod.EVA):
        assert len(parameters["participants"]) * parameters["local_classes"] == sum(parameters["nodes_per_class"])

        self.parameters = parameters
        self.data_dir = data_dir
        self.sample_size = parameters["sample_size"]
        self.model = create_model(parameters["dataset"], parameters["model"])
        self.participants = parameters["participants"]
        self.logger.info("Setting up experiment with %d participants and sample size %d (I am participant %d)" %
                         (len(self.participants), self.sample_size, self.get_my_participant_index()))

        self.dataset = Dataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters, self.get_my_participant_index())
        self.optimizer = SGDOptimizer(self.model, parameters["learning_rate"], parameters["momentum"])

        # Setup the model transmission
        self.transmission_method = transmission_method
        if self.transmission_method == TransmissionMethod.EVA:
            self.logger.info("Setting up EVA protocol")
            self.eva_init(window_size_in_blocks=32, retransmit_attempt_count=10, retransmit_interval_in_sec=1, timeout_interval_in_sec=10)
            self.eva_register_receive_callback(self.on_receive)
            self.eva_register_send_complete_callback(self.on_send_complete)
            self.eva_register_error_callback(self.on_error)

    def train(self) -> bool:
        """
        Train the model on a batch. Return a boolean that indicates whether the epoch is completed.
        """
        def it_has_next(iterable):
            try:
                first = next(iterable)
            except StopIteration:
                return None
            return itertools.chain([first], iterable)

        data, target = self.dataset.iterator.__next__()
        self.model.train()
        data, target = Variable(data), Variable(target)
        self.optimizer.optimizer.zero_grad()
        self.logger.info('d-sgd.next node forward propagation')
        output = self.model.forward(data)
        loss = F.nll_loss(output, target)
        self.logger.info('d-sgd.next node backward propagation')
        loss.backward()
        self.optimizer.optimizer.step()

        # Are we at the end of the epoch?
        res = it_has_next(self.dataset.iterator)
        if res is None:
            self.epoch += 1
            self.logger.info("Epoch done - resetting dataset iterator")
            self.dataset.reset_train_iterator()
            return True
        else:
            self.dataset.iterator = res
            return False

    def average_models(self, models):
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model

    async def send_aggregated_model(self, round, model):
        """
        Send the global model update to the participants of the next round.
        """
        participants_next_round = self.get_participants_for_round(round + 1)
        for participant_ind in participants_next_round:
            if participant_ind == self.get_my_participant_index():
                continue

            participant_pk = unhexlify(self.participants[participant_ind])
            peer = self.get_peer_by_pk(participant_pk)
            if not peer:
                self.logger.warning("Peer object of participant %d not available - not sending aggregated model for round %d", participant_ind, round)
                continue

            if self.transmission_method == TransmissionMethod.EVA:
                await self.eva_send_aggregated_model(round, model, peer)

    async def eva_send_aggregated_model(self, round, model, peer):
        response = {"round": round, "type": "aggregated_model"}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            self.logger.info("Participant %d sending round %d aggregated model to participant %d (attempt %d)",
                             self.get_my_participant_index(), round,
                             self.get_participant_index(peer.public_key.key_to_bin()), attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(model))
                self.logger.info("Participant %d successfully sent aggregated model of round %d to participant %s",
                                 self.get_my_participant_index(), round,
                                 self.get_participant_index(peer.public_key.key_to_bin()))
                break
            except Exception:
                self.logger.exception("Exception when sending aggregated model to peer %s", peer)
            attempt += 1

    async def send_local_model(self):
        """
        Send the global model to the round representative.
        """
        if self.is_round_representative(self.round):
            return

        round_representative = self.get_round_representative(self.round)
        participant_pk = unhexlify(self.participants[round_representative])
        peer = self.get_peer_by_pk(participant_pk)
        if not peer:
            self.logger.warning("Peer object of round representative %d not available - not sending local model", round_representative)
            return

        if self.transmission_method == TransmissionMethod.EVA:
            await self.eva_send_local_model(peer)

    async def eva_send_local_model(self, peer):
        response = {"round": self.round, "type": "local_model"}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            if self.model_send_delay is not None:
                await sleep(random.randint(0, self.model_send_delay) / 1000)
            self.logger.info("Participant %d sending round %d local model to participant %d (attempt %d)",
                             self.get_my_participant_index(), self.round,
                             self.get_participant_index(peer.public_key.key_to_bin()), attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(self.model))
                self.logger.info("Local model successfully sent to participant %d",
                                 self.get_participant_index(peer.public_key.key_to_bin()))
                break
            except Exception:
                self.logger.exception("Exception when sending model to peer %s", peer)
            attempt += 1

    def adopt_model(self, new_model):
        """
        Replace the parameters of the current model with those of a new model.
        """
        with torch.no_grad():
            for p, new_p in zip(self.model.parameters(), new_model.parameters()):
                p.mul_(0.)
                p.add_(new_p)

    async def participate_in_round(self):
        """
        Complete a round of training and model aggregation.
        """
        self.is_participating_in_round = True
        self.logger.info("Participant %d starts participating in round %d", self.get_my_participant_index(), self.round)

        # It can happen that this node is still computing the accuracy of the model produced by the previous round
        # when starting the next round. If so, we wait until this accuracy computation is done.
        if self.is_computing_accuracy:
            self.logger.info("Waiting for accuracy computation to finish")
            await self.compute_accuracy_deferred

        # Adopt the aggregated model sent by other nodes
        # TODO there can be inconsistencies in the models received - assume for now they are all the same
        if self.round > 1:
            # Replace the current model with the received aggregated model
            self.adopt_model(random.choice(self.incoming_aggregated_models[self.round - 1]))
            self.incoming_aggregated_models.pop(self.round - 1, None)

        # Train
        epoch_done = self.train()

        await self.send_local_model()

        avg_model = self.model
        if self.sample_size > 1:
            if self.is_round_representative(self.round) and ((self.round not in self.incoming_local_models) or (self.round in self.incoming_local_models and len(self.incoming_local_models[self.round]) < self.sample_size - 1)):
                await self.round_deferred
                self.logger.info("Round representative %d received %d model(s) from other peers for round %d - "
                                 "starting to average", self.get_my_participant_index(),
                                 len(self.incoming_local_models[self.round]), self.round)

                # Average your model with those of the other participants
                avg_model = self.average_models(self.incoming_local_models[self.round] + [self.model])
                with torch.no_grad():
                    for p, new_p in zip(self.model.parameters(), avg_model.parameters()):
                        p.mul_(0.)
                        p.add_(new_p)

        if self.is_round_representative(self.round):
            if self.round not in self.incoming_aggregated_models:
                self.incoming_aggregated_models[self.round] = []
            self.incoming_aggregated_models[self.round].append(copy.deepcopy(avg_model))
            self.register_task("send_aggregated_model_%d" %
                               self.round, self.send_aggregated_model, self.round, avg_model)

        self.incoming_local_models.pop(self.round, None)
        self.round_deferred = Future()

        self.logger.info("Participant %d finished round %d", self.get_my_participant_index(), self.round)
        if self.round_complete_callback:
            await self.round_complete_callback(self.round, epoch_done)
        self.is_participating_in_round = False

        # Should I participate in the next round again?
        if self.sample_size == 1 and self.is_participant_for_round(self.round + 1):
            self.round += 1
            ensure_future(self.participate_in_round())

        # Check if there is a future round in which we participate and for which we have received all models.
        # If so, start participating in that round.
        for round_nr in self.incoming_aggregated_models:
            if round_nr < self.round or not self.is_participant_for_round(round_nr + 1):
                continue
            if len(self.incoming_aggregated_models[round_nr]) == 1:
                self.round = round_nr + 1
                ensure_future(self.participate_in_round())

    async def compute_accuracy(self, include_wait_periods=True):
        """
        Compute the accuracy/loss of the current model.
        Optionally, one can provide a custom iterator to compute the accuracy on a custom dataset.
        """
        self.logger.info("Computing accuracy of model")
        self.is_computing_accuracy = True
        self.compute_accuracy_deferred = Future()
        correct = example_number = total_loss = num_batches = 0
        copied_model = copy.deepcopy(self.model)
        copied_model.eval()

        with torch.no_grad():
            for data, target in self.dataset.test_iterator:
                output = copied_model.forward(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1
                if include_wait_periods:
                    await sleep(0.001)

        accuracy = float(correct) / float(example_number)
        loss = total_loss / float(example_number)
        self.logger.info("Finished computing accuracy of model (accuracy: %f, loss: %f)", accuracy, loss)
        self.dataset.reset_test_iterator()
        self.is_computing_accuracy = False
        self.compute_accuracy_deferred.set_result(None)
        return accuracy, loss

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    def received_local_model(self, participant: int, model_round: int, serialized_model: bytes) -> None:
        self.logger.info("Received local model for round %d from participant %d", model_round, participant)
        if model_round == self.round:
            if not self.is_round_representative(model_round):
                self.logger.warning("We received a local model for round %d from participant %d but we are "
                                    "not the round representative" % model_round, participant)
                return

            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            if model_round not in self.incoming_local_models:
                self.incoming_local_models[model_round] = []
            self.incoming_local_models[model_round].append(incoming_model)
            self.logger.info("Received expected local model (now have %d/%d)",
                             len(self.incoming_local_models[self.round]), self.sample_size - 1)
            if len(self.incoming_local_models[self.round]) == self.sample_size - 1 and not self.round_deferred.done():
                self.round_deferred.set_result(None)
        elif model_round > self.round and self.is_participant_for_round(model_round):
            self.logger.info("Participant %d received a local model from participant %d for future round %d",
                             self.get_my_participant_index(), participant, model_round)
            # It is possible that we receive a model for a later round while we are still in an earlier round.
            if model_round not in self.incoming_local_models:
                self.incoming_local_models[model_round] = []
            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            self.incoming_local_models[model_round].append(incoming_model)
        else:
            self.logger.warning("Participant %d received a local model from participant %d for a round (%d) that is "
                                "not relevant for us (we're in round %d)",
                                self.get_my_participant_index(), participant, model_round, self.round)

    def received_aggregated_model(self, participant: int, model_round: int, serialized_model: bytes) -> None:
        if not self.is_participant_for_round(model_round + 1):
            self.logger.warning("Received aggregated model from participant %d for round %d but we are not a "
                                "participant in that round", participant, model_round + 1)

        self.logger.info("Received aggregated model for round %d from participant %d", model_round, participant)
        incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
        if model_round not in self.incoming_aggregated_models:
            self.incoming_aggregated_models[model_round] = []
        self.incoming_aggregated_models[model_round].append(incoming_model)
        if len(self.incoming_aggregated_models[model_round]) == 1 and not self.is_participating_in_round:
            # Perform this round
            self.round = model_round + 1
            ensure_future(self.participate_in_round())

    def on_receive(self, result: TransferResult):
        participant_index = self.get_participant_index(result.peer.public_key.key_to_bin())
        self.logger.info(f'Participant {self.get_my_participant_index()} received data from participant '
                         f'{participant_index}: {result.info.decode()}')
        json_data = json.loads(result.info.decode())
        if json_data["type"] == "aggregated_model":
            participant = self.get_participant_index(result.peer.public_key.key_to_bin())
            self.received_aggregated_model(participant, json_data["round"], result.data)
        elif json_data["type"] == "local_model":
            participant = self.get_participant_index(result.peer.public_key.key_to_bin())
            self.received_local_model(participant, json_data["round"], result.data)

    def on_send_complete(self, result: TransferResult):
        participant_ind = self.get_participant_index(result.peer.public_key.key_to_bin())
        self.logger.info(f'Outgoing transfer to participant {participant_ind} has completed: {result.info.decode()}')

    def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')
