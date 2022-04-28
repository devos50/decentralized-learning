from binascii import hexlify
from typing import List, Dict

NEVER_ACTIVE = -1


class PeerManager:
    """
    The PeerManager keeps track of the population of peers that are participating in the training process.
    """

    def __init__(self, my_pk: bytes):
        self.my_pk = my_pk
        self.peers: List[bytes] = []
        self.last_active: Dict[bytes, int] = {}

    def add_peer(self, peer_pk: bytes) -> None:
        """
        Add a new peer to this manager.
        :param peer_pk: The public key of the peer to add.
        """
        # TODO we assume here that the peer is eligible for participation in the process.
        self.peers.append(peer_pk)
        self.last_active[peer_pk] = NEVER_ACTIVE

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
