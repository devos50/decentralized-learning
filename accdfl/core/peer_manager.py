import pickle
from binascii import hexlify
from typing import List, Dict, Optional, Tuple

from accdfl.core import NodeDelta

NO_ACTIVITY_INFO = -1


class PeerManager:
    """
    The PeerManager keeps track of the population of peers that are participating in the training process.
    """

    def __init__(self, my_pk: bytes):
        self.my_pk = my_pk
        self.peers: List[bytes] = []
        self.last_active: Dict[bytes, int] = {}
        self.node_deltas: List[Tuple[bytes, NodeDelta, int]] = []

    def add_peer(self, peer_pk: bytes, round_active: Optional[int] = NO_ACTIVITY_INFO) -> None:
        """
        Add a new peer to this manager.
        :param peer_pk: The public key of the peer to add.
        :param round_active: The round that we last heard from this peer.
        """
        if peer_pk in self.peers:
            return

        self.peers.append(peer_pk)
        self.last_active[peer_pk] = round_active

    def remove_peer(self, peer_pk) -> None:
        """
        Remove this peer from this manager.
        :param peer_pk: The public key of the peer to remove.
        """
        if peer_pk in self.peers:
            self.peers.remove(peer_pk)
        self.last_active.pop(peer_pk, None)

    def update_peer_activity(self, peer_pk: bytes, round_active) -> None:
        """
        Update the status of a particular peer.
        :param peer_pk: The public key of the peer which activity status will be udpated.
        :param round_active: The last round in which this peer has been active.
        """
        self.last_active[peer_pk] = max(self.last_active[peer_pk], round_active)

    def get_my_short_id(self) -> str:
        """
        Return a short description of your public key
        """
        return hexlify(self.my_pk).decode()[-8:]

    @staticmethod
    def get_short_id(peer_pk: bytes) -> str:
        return hexlify(peer_pk).decode()[-8:]

    def peer_is_in_node_deltas(self, peer_pk: bytes) -> bool:
        for pk, _, __ in self.node_deltas:
            if pk == peer_pk:
                return True
        return False

    def update_node_deltas(self, round: int, serialized_node_deltas: bytes) -> None:
        node_deltas = [(pk, NodeDelta(delta), ttl) for pk, delta, ttl in pickle.loads(serialized_node_deltas)]
        for peer_pk, delta, _ in node_deltas:
            if peer_pk not in self.peers and delta == NodeDelta.JOIN:
                self.add_peer(peer_pk, round_active=round)
            elif peer_pk in self.peers and delta == NodeDelta.LEAVE:
                self.remove_peer(peer_pk)

        # Decrement TTLs and ignore entries which TTL will be zero
        self.node_deltas = [(pk, delta, ttl - 1) for pk, delta, ttl in node_deltas if ttl > 1]

    def get_serialized_node_deltas(self) -> bytes:
        raw_list = [(pk, delta.value(), ttl) for pk, delta, ttl in self.node_deltas]
        return pickle.dumps(raw_list)
