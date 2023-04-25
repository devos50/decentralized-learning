import asyncio
import copy
import json
import random
import time
from asyncio import sleep, ensure_future
from binascii import unhexlify
from typing import List

from accdfl.core.community import LearningCommunity
from accdfl.core.models import serialize_model, unserialize_model
from accdfl.util.eva.result import TransferResult


class GLCommunity(LearningCommunity):
    community_id = unhexlify('e5889074c1e4c60423cdb6e9307ba0ca56a5ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round: int = 1
        self.model_age: int = 0
        self.nodes = None
        self.round_task_name = None

    def start(self):
        """
        Start to participate in the training process.
        """
        super().start()
        self.start_next_round()

    def go_offline(self, graceful: bool = True):
        super().go_offline()
        if self.round_task_name and self.is_pending_task_active(self.round_task_name):
            self.logger.info("Participant %s cancelling round task %s",
                             self.peer_manager.get_my_short_id(), self.round_task_name)
            self.cancel_pending_task(self.round_task_name)
            self.round_task_name = None

    def go_online(self):
        super().go_online()
        self.start_next_round()

    def eva_send_model(self, round: int, model_age: int, model, peer):
        start_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        serialized_model = serialize_model(model)
        response = {"round": round, "model_age": model_age}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, serialized_model, start_time)

    def start_next_round(self):
        self.round_task_name = "round_%d" % self.round
        if not self.is_pending_task_active(self.round_task_name):
            self.register_task(self.round_task_name, self.do_round)
        else:
            self.logger.warning("Task %s of participant %s already seems to be active!",
                                self.round_task_name, self.peer_manager.get_my_short_id())

    async def do_round(self):
        """
        Perform a single round.
        """
        self.logger.info("Peer %s starting round %d", self.peer_manager.get_my_short_id(), self.round)

        # Wait
        await sleep(self.settings.gl.round_timeout)

        # Select a random neighbour and send the model.
        online_nodes: List = [node for node in self.nodes if node.overlays[0].is_active]
        if online_nodes:
            rand_online_node = random.choice(online_nodes)
            peer = rand_online_node.overlays[0].my_peer

            await self.eva_send_model(self.round, self.model_age, self.model_manager.model, peer)
        else:
            self.logger.warning("Peer %s has no neighbouring online peer, skipping model transfer!",
                                self.peer_manager.get_my_short_id())

        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(self.round))
        self.logger.info("Peer %s completed round %d", self.peer_manager.get_my_short_id(), self.round)
        self.round += 1

        self.round_task_name = None

        if self.is_active:
            self.round_task_name = "round_%d" % self.round
            self.register_task(self.round_task_name, self.do_round)

    async def on_receive(self, result: TransferResult):
        """
        We received a model from a neighbouring peer.
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

        # Merge the incoming model with the current local model.
        detached_model = unserialize_model(serialize_model(self.model_manager.model),
                                           self.settings.dataset, architecture=self.settings.model)
        self.model_manager.process_incoming_trained_model(self.my_peer.public_key.key_to_bin(), detached_model)
        self.model_manager.process_incoming_trained_model(peer_pk, incoming_model)

        age_sum = json_data["model_age"] + self.model_age
        weights = [self.model_age / age_sum, json_data["model_age"] / age_sum] if age_sum > 0 else None
        self.model_age = max(json_data["model_age"], self.model_age)
        self.logger.info("Aggregating local and remote model with weights: %s", weights)
        self.model_manager.model = self.model_manager.aggregate_trained_models(weights=weights)
        self.model_manager.reset_incoming_trained_models()
        if self.aggregate_complete_callback:
            model_cpy = copy.deepcopy(self.model_manager.model)
            ensure_future(self.aggregate_complete_callback(self.round, model_cpy))

        # Train
        self.model_age += await self.model_manager.train()
