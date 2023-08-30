import asyncio
import copy
import json
import time
from asyncio import ensure_future
from binascii import unhexlify
from typing import List, Tuple

import torch
from torch import nn

from accdfl.core.community import LearningCommunity
from accdfl.core.models import serialize_model, unserialize_model
from accdfl.util.eva.result import TransferResult


class DLCommunity(LearningCommunity):
    community_id = unhexlify('e5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round: int = 0
        self.neighbours: List[bytes] = []  # The PKs of the neighbours we will send our model to
        self.incoming_models: List[Tuple[bytes, nn.Module]] = []  # Incoming models for a round

    def start(self):
        """
        Start to participate in the training process.
        """
        super().start()
        if not self.neighbours:
            raise RuntimeError("No neighbours for peer %s", self.peer_manager.get_my_short_id())

    def go_offline(self, graceful: bool = True):
        super().go_offline(graceful=graceful)
        train_task_name = "round_%d" % self.round
        if self.is_pending_task_active(train_task_name):
            self.logger.warning("Cancelling training task of participant %s as it goes offline",
                                self.peer_manager.get_my_short_id())
            self.cancel_pending_task(train_task_name)

    def eva_send_model(self, round, model, peer):
        start_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        serialized_model = serialize_model(model)
        response = {"round": round}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, serialized_model, start_time)

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

        # Detach the tensors of the model by making a copy
        model_cpy = unserialize_model(serialize_model(self.model_manager.model),
                                      self.settings.dataset, architecture=self.settings.model)

        my_peer_pk = self.my_peer.public_key.key_to_bin()
        self.incoming_models.append((my_peer_pk, model_cpy))

        # Send the trained model to your neighbours
        to_send = self.neighbours
        if self.settings.dl.topology == "exp-one-peer":
            nb_ind = (self.round - 1) % len(self.neighbours)
            to_send = [self.neighbours[nb_ind]]

        for peer_pk in to_send:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Participant %s cannot find Peer object for participant %s!",
                                    self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk))
                continue

            self.logger.info("Participant %s sending model of round %d to participant %s",
                             self.peer_manager.get_my_short_id(), self.round,
                             self.peer_manager.get_short_id(peer.public_key.key_to_bin()))
            ensure_future(self.eva_send_model(self.round, self.model_manager.model, peer))

    def aggregate_models(self):
        """
        Aggregate the received models.
        """
        if not self.incoming_models:
            # Nothing to aggregate
            return

        # The round is complete - wrap it up and proceed
        self.logger.info("Participant %s received %d models, aggregating...",
                         self.peer_manager.get_my_short_id(), len(self.incoming_models))
        self.model_manager.incoming_trained_models = dict((x, y) for x, y in self.incoming_models)

        # Transfer these models back to the CPU to prepare for aggregation
        device = torch.device("cpu")
        for peer_pk in self.model_manager.incoming_trained_models.keys():
            model = self.model_manager.incoming_trained_models[peer_pk]
            self.model_manager.incoming_trained_models[peer_pk] = model.to(device)

        self.model_manager.model = self.model_manager.aggregate_trained_models()
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(self.round))
        if self.aggregate_complete_callback:
            model_cpy = copy.deepcopy(self.model_manager.model)
            ensure_future(self.aggregate_complete_callback(self.round, model_cpy))
        self.logger.error("Peer %s completed round %d", self.peer_manager.get_my_short_id(), self.round)
        self.incoming_models = []

    async def on_receive(self, result: TransferResult):
        """
        We received a model from a neighbouring peer. Store it and check if we received enough models to proceed.
        """
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.debug("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')

        json_data = json.loads(result.info.decode())
        incoming_model = unserialize_model(result.data, self.settings.dataset, architecture=self.settings.model)
        self.process_incoming_model(incoming_model, peer_pk)

    def process_incoming_model(self, incoming_model: nn.Module, peer_pk: bytes):
        self.incoming_models.append((peer_pk, incoming_model))
