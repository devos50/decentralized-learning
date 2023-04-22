from asyncio import sleep, ensure_future, Future
from typing import Optional, List, Tuple

from accdfl.dfl.community import DFLCommunity
from accdfl.util.eva.result import TransferResult
from ipv8.types import Peer


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.bandwidth: Optional[float] = None
        self.transfers: List[Tuple[str, str, int, float, bool, str]] = []

    def schedule_eva_send_model(self, peer: Peer, serialized_response: bytes, binary_data: bytes, start_time: float) -> Future:
        # Schedule the transfer
        future = ensure_future(self.bypass_send(peer, serialized_response, binary_data))
        future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, serialized_response, binary_data, start_time))
        return future

    async def bypass_send(self, peer: Peer, serialized_response: bytes, binary_data: bytes):
        found: bool = False
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                self.endpoint.bytes_up += len(binary_data) + len(serialized_response)
                node.overlays[0].endpoint.bytes_down += len(binary_data) + len(serialized_response)

                if self.bandwidth:
                    transfer_size_kbits = (len(binary_data) + len(serialized_response)) / 1024 * 8
                    transfer_time = transfer_size_kbits / self.bandwidth
                    await sleep(transfer_time)
                    self.logger.info("Model transfer took %f s.", transfer_time)

                if not node.overlays[0].is_active:  # The node might have gone offline at this point in time
                    break

                res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)
