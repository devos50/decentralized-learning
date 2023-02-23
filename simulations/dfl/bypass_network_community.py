from asyncio import ensure_future, sleep, get_event_loop

from accdfl.core.models import unserialize_model, serialize_model
from accdfl.dfl.community import DFLCommunity


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.model_send_times = {}

    async def eva_send_model(self, round, model, type, population_view, peer):
        model_cpy = unserialize_model(serialize_model(self.model_manager.model),
                                      self.settings.dataset, architecture=self.settings.model)
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                peer_pk = peer.public_key.key_to_bin()

                # Simulate the time it takes for a transfer
                if peer_pk in self.model_send_times:
                    await sleep(self.model_send_times[peer_pk])

                    node.overlays[0].peer_manager.merge_population_views(population_view)
                    node.overlays[0].peer_manager.update_peer_activity(self.my_peer.public_key.key_to_bin(),
                                                                       max(round, self.get_round_estimate()))
                    node.overlays[0].update_population_view_history()

                    if type == "trained_model":
                        ensure_future(node.overlays[0].received_trained_model(self.my_peer, round, model_cpy))
                    elif type == "aggregated_model":
                        node.overlays[0].received_aggregated_model(self.my_peer, round, model_cpy)
                else:
                    # Measure once the time it takes to do a model transfer
                    start_time = get_event_loop().time()
                    await super().eva_send_model(round, model, type, population_view, peer)
                    transfer_time = get_event_loop().time() - start_time
                    latency_out = self.endpoint.latencies[node.overlays[0].endpoint.wan_address]
                    latency_in = node.overlays[0].endpoint.latencies[self.endpoint.wan_address]
                    self.logger.info("Model transfer time from peer %s to %s established to be %f s. "
                                     "(latency out: %f, latency in: %f)",
                                     self.peer_manager.get_my_short_id(),
                                     self.peer_manager.get_short_id(peer_pk), transfer_time, latency_out, latency_in)
                    self.model_send_times[peer_pk] = transfer_time

                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
