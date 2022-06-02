import random
import time
from asyncio import sleep, ensure_future
from binascii import unhexlify

from accdfl.core.community import DLCommunity
from accdfl.core.models import serialize_model, unserialize_model
from accdfl.util.eva.result import TransferResult

from ipv8.peer import Peer


class GLCommunity(DLCommunity):
    """
    Community implementing Gossip Learning.
    See http://publicatio.bibl.u-szeged.hu/15824/1/dais19a.pdf
    """
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0
        self.logger.info("The GL community started with peer ID: %s", self.peer_manager.get_my_short_id())

    def get_random_peer(self) -> Peer:
        return random.choice(self.get_peers())

    def eva_send_model(self, model, peer: Peer):
        start_time = time.time()
        serialized_model = serialize_model(model)
        return self.schedule_eva_send_model(peer, b"", serialized_model, start_time)

    def start(self):
        self.round = 1
        self.is_active = True
        delay = random.random() * self.settings.gl.round_duration
        self.logger.info("Participant %s starts training - delaying main loop start for %f s.",
                         self.peer_manager.get_my_short_id(), delay)
        self.register_task("main_loop", self.loop, delay=delay)

    async def loop(self):
        while True:
            await sleep(self.settings.gl.round_duration)
            if self.get_peers():
                random_peer = self.get_random_peer()
                self.logger.info("Sending local model to participant %s",
                                 self.peer_manager.get_short_id(random_peer.public_key.key_to_bin()))
                ensure_future(self.eva_send_model(self.model_manager.model, random_peer))

    async def on_receive(self, result: TransferResult):
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()

        if not self.is_active:
            self.logger.warning("Participant %s ignoring message from %s due to inactivity", my_peer_id, peer_id)
            return

        self.logger.info(f'Participant {my_peer_id} received model from participant {peer_id}: {result.info.decode()}')
        incoming_model = unserialize_model(result.data, self.settings.dataset)

        # Merge the incoming model with your model
        self.model_manager.model = self.model_manager.average_models([self.model_manager.model, incoming_model])
        await self.model_manager.train()
        if self.aggregate_complete_callback:
            ensure_future(self.aggregate_complete_callback(self.round, self.model_manager.model))
        self.round += 1
