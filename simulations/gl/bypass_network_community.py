import asyncio
import json
from asyncio import sleep, ensure_future, Future
from typing import Optional, List, Tuple

from accdfl.gl.community import GLCommunity
from accdfl.util.eva.result import TransferResult
from ipv8.types import Peer


class GLBypassNetworkCommunity(GLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bandwidth: Optional[float] = None
        self.available_for_send: float = 0
        self.available_for_receive: float = 0
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []
        self.total_time_sending: float = 0
        self.total_time_receiving: float = 0

    def schedule_eva_send_model(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float) -> Future:
        # Schedule the transfer
        future = ensure_future(self.bypass_send(peer, serialized_response, binary_data))
        future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, serialized_response, binary_data, start_time))
        return future

    async def bypass_send(self, peer: Peer, serialized_response: bytes, binary_data: bytes):
        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        transfer_wait_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                self.endpoint.bytes_up += len(binary_data) + len(serialized_response)
                node.overlays[0].endpoint.bytes_down += len(binary_data) + len(serialized_response)

                if self.bandwidth:
                    transfer_size_kbits = (len(binary_data) + len(serialized_response)) / 1024 * 8
                    transfer_time = transfer_size_kbits / min(self.bandwidth, node.overlays[0].bandwidth)

                    # Schedule the transfer between the two nodes.
                    # Simply schedule it at the max. end time in the transfer queue of both the sender and receiver.
                    cur_time = asyncio.get_event_loop().time()
                    earliest_start_time_sender = cur_time if self.available_for_send <= cur_time else self.available_for_send
                    earliest_start_time_receiver = cur_time if node.overlays[0].available_for_receive <= cur_time else node.overlays[0].available_for_receive
                    transfer_start_time = max(earliest_start_time_sender, earliest_start_time_receiver)
                    transfer_wait_time = transfer_start_time - cur_time

                    # Update the earliest available time of both nodes
                    self.available_for_send = transfer_start_time + transfer_time
                    node.overlays[0].available_for_receive = transfer_start_time + transfer_time

                    await sleep(transfer_wait_time + transfer_time)
                    self.logger.info("Model transfer %s => %s started at t=%f and took %f s. (waiting time: %f s.)",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time, transfer_time, transfer_wait_time)
                    self.total_time_sending += transfer_time
                    self.total_time_receiving += transfer_time

                    # The transfer only succeeds if both nodes are online when the transfer is done
                    transfer_success = node.overlays[0].is_active and self.is_active

                json_data = json.loads(serialized_response.decode())
                self.transfers.append((self.peer_manager.get_my_short_id(),
                                       node.overlays[0].peer_manager.get_my_short_id(), json_data["round"],
                                       transfer_wait_time, transfer_time, json_data["type"], transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
