import copy
from asyncio import ensure_future

from accdfl.core.models import unserialize_model, serialize_model
from accdfl.dfl.community import DFLCommunity


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None

    async def eva_send_model(self, round, model, type, population_view, peer):
        model_cpy = unserialize_model(serialize_model(self.model_manager.model),
                                      self.settings.dataset, architecture=self.settings.model)
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True

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
