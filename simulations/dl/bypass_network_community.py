import copy

from accdfl.dl.community import DLCommunity


class DLBypassNetworkCommunity(DLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None

    async def eva_send_model(self, round, model, peer):
        model_cpy = copy.deepcopy(model)
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                node.overlays[0].process_incoming_model(model_cpy, peer.public_key.key_to_bin(), round)
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
