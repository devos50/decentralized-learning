from binascii import hexlify
from typing import List, Dict, Optional

NO_ACTIVITY_INFO = -1


class PeerManager:
    """
    The PeerManager keeps track of the population of peers that are participating in the training process.
    """

    def __init__(self, my_pk: bytes):
        self.my_pk = my_pk
        self.peers: List[bytes] = []
        self.last_active: Dict[bytes, int] = {}

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
