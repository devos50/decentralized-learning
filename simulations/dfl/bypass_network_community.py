import asyncio
import json
import random
from asyncio import ensure_future, Future
from typing import List, Tuple, Set

from accdfl.dfl.community import DFLCommunity
from accdfl.util.eva.result import TransferResult
from ipv8.types import Peer
from simulations.bandwidth_scheduler import BWScheduler


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.available_for_send: float = 0
        self.available_for_receive: float = 0
        self.transfers: List[Tuple[str, str, int, float, str, bool]] = []

        self.bw_scheduler: BWScheduler = BWScheduler(self.peer_manager.get_my_short_id())

    def schedule_eva_send_model(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float) -> Future:
        # Schedule the transfer
        future = ensure_future(self.bypass_send(peer, serialized_response, binary_data))
        future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, serialized_response, binary_data, start_time))
        return future

    def go_offline(self, graceful: bool = True) -> None:
        super().go_offline(graceful=graceful)
        self.bw_scheduler.kill_all_transfers()

    async def bypass_send(self, peer: Peer, serialized_response: bytes, binary_data: bytes):
        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                if self.bw_scheduler.bw_limit > 0:
                    transfer_size_kbits = (len(binary_data) + len(serialized_response)) / 1024 * 8
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size_kbits)
                    transfer_start_time = asyncio.get_event_loop().time()
                    try:
                        await transfer.complete_future
                    except RuntimeError:
                        transfer_success = False
                    transfer_time = asyncio.get_event_loop().time() - transfer_start_time

                    self.endpoint.bytes_up += transfer.get_transferred_bytes()
                    node.overlays[0].endpoint.bytes_down += transfer.get_transferred_bytes()

                    self.logger.info("Model transfer %s => %s started at t=%f and took %f s.",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time, transfer_time)
                else:
                    self.endpoint.bytes_up += len(binary_data) + len(serialized_response)
                    node.overlays[0].endpoint.bytes_down += len(binary_data) + len(serialized_response)

                json_data = json.loads(serialized_response.decode())
                self.transfers.append((self.peer_manager.get_my_short_id(),
                                       node.overlays[0].peer_manager.get_my_short_id(), json_data["round"],
                                       transfer_time, json_data["type"], transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
