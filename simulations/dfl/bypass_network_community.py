import pickle
from asyncio import ensure_future, sleep, get_event_loop
from typing import Optional

from accdfl.core.models import unserialize_model, serialize_model
from accdfl.dfl.community import DFLCommunity


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.bandwidth: Optional[float] = None

    async def eva_send_model(self, round, model, type, population_view, peer):
        serialized_model = serialize_model(self.model_manager.model)
        serialized_population_view = pickle.dumps(population_view)
        transfer_size_kbits = (len(serialized_model) + len(serialized_population_view)) / 1024 * 8
        model_cpy = unserialize_model(serialized_model, self.settings.dataset, architecture=self.settings.model)
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True

                if not node.overlays[0].is_active:
                    break

                # Simulate the time it takes for a transfer
                if self.bandwidth:
                    transfer_time = transfer_size_kbits / self.bandwidth
                    await sleep(transfer_time)
                    self.logger.info("Model transfer took %f s.", transfer_time)

                node.overlays[0].peer_manager.merge_population_views(population_view)
                node.overlays[0].peer_manager.update_peer_activity(self.my_peer.public_key.key_to_bin(),
                                                                   max(round, self.get_round_estimate()))
                node.overlays[0].update_population_view_history()

                if type == "trained_model":
                    ensure_future(node.overlays[0].received_trained_model(self.my_peer, round, model_cpy))
                elif type == "aggregated_model":
                    node.overlays[0].received_aggregated_model(self.my_peer, round, model_cpy)

                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
