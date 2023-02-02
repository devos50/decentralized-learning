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
        self.neighbours: List[bytes] = []  # The PKs of the neighbours we will send our model to

    def start(self):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"
        assert self.neighbours, "We need some neighbours"
        self.start_next_round()

    def eva_send_model(self, round: int, model_age: int, model, peer):
        start_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        serialized_model = serialize_model(model)
        response = {"round": round, "model_age": model_age}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, serialized_model, start_time)

    def start_next_round(self):
        self.register_task("round_%d" % self.round, self.do_round)

    async def do_round(self):
        """
        Perform a single round.
        """
        self.logger.info("Peer %s starting round %d", self.peer_manager.get_my_short_id(), self.round)

        # Wait
        await sleep(self.settings.gl.round_timeout)

        # Select a random neighbour and send the model
        peer_pk = random.choice(self.neighbours)
        peer = self.get_peer_by_pk(peer_pk)
        if not peer:
            raise RuntimeError("Participant %s cannot find Peer object for participant %s!" % (
                               self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk)))

        await self.eva_send_model(self.round, self.model_age, self.model_manager.model, peer)

        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(self.round))
        self.logger.info("Peer %s completed round %d", self.peer_manager.get_my_short_id(), self.round)
        self.round += 1
        self.register_task("round_%d" % self.round, self.do_round)

    async def on_receive(self, result: TransferResult):
        """
        We received a model from a neighbouring peer.
        """
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()
        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')

        json_data = json.loads(result.info.decode())
        incoming_model = unserialize_model(result.data, self.settings.dataset, architecture=self.settings.model)

        # Merge the incoming model with the current local model.
        self.model_manager.process_incoming_trained_model(peer_pk, incoming_model)

        age_sum = json_data["model_age"] + self.model_age
        weights = [self.model_age / age_sum, json_data["model_age"] / age_sum] if age_sum > 0 else None
        self.model_age = max(json_data["model_age"], self.model_age)
        self.model_manager.model = self.model_manager.aggregate_trained_models(weights=weights)
        self.model_manager.reset_incoming_trained_models()
        if self.aggregate_complete_callback:
            model_cpy = copy.deepcopy(self.model_manager.model)
            ensure_future(self.aggregate_complete_callback(self.round, model_cpy))

        # Train
        self.model_age += await self.model_manager.train()
