import asyncio
import json
import pickle
from asyncio import sleep, ensure_future
from typing import Optional, List, Tuple

from accdfl.core.models import unserialize_model, serialize_model
from accdfl.dfl.community import DFLCommunity
from accdfl.util.eva.result import TransferResult


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.bandwidth: Optional[float] = None
        self.transfers: List[Tuple[str, str, int, float, bool, str]] = []

    async def eva_send_model(self, round, model, type, population_view, peer):
        serialized_model = serialize_model(self.model_manager.model)
        serialized_population_view = pickle.dumps(population_view)
        transfer_size_kbits = (len(serialized_model) + len(serialized_population_view)) / 1024 * 8
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

                success: bool = False
                if node.overlays[0].is_active:
                    success = True
                    self.endpoint.bytes_up += len(serialized_model) + len(serialized_population_view)
                    node.overlays[0].endpoint.bytes_down += len(serialized_model) + len(serialized_population_view)

                    info = {"round": round, "type": type, "model_data_len": len(serialized_model)}
                    transfer = TransferResult(self.my_peer, json.dumps(info).encode(),
                                              serialized_model + serialized_population_view, 0)
                    ensure_future(node.overlays[0].on_receive(transfer))

                peer_pk = node.overlays[0].my_peer.public_key.key_to_bin()
                cur_time = asyncio.get_event_loop().time()
                type_short = type.split("_")[0]
                self.transfers.append((self.peer_manager.get_my_short_id(), self.peer_manager.get_short_id(peer_pk), round, cur_time, success, type_short))

                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
