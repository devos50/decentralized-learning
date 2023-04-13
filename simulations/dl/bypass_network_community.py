from accdfl.core.models import unserialize_model, serialize_model
from accdfl.dl.community import DLCommunity


class DLBypassNetworkCommunity(DLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None

    async def eva_send_model(self, round, model, peer):
        model_cpy = unserialize_model(serialize_model(self.model_manager.model),
                                      self.settings.dataset, architecture=self.settings.model)
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if node.overlays[0].is_active:
                    node.overlays[0].process_incoming_model(model_cpy, peer.public_key.key_to_bin(), round)
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
