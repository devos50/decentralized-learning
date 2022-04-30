import logging
from binascii import hexlify
from typing import Dict, Optional, List

NO_ACTIVITY_INFO = -1
WENT_OFFLINE = -2


class PeerManager:
    """
    The PeerManager keeps track of the population of peers that are participating in the training process.
    """

    def __init__(self, my_pk: bytes):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_pk = my_pk
        self.last_active: Dict[bytes, int] = {}
        self.last_active_pending: Dict[bytes, int] = {}  # Pending changes that are not immediately applied.

    def add_peer(self, peer_pk: bytes, round_active: Optional[int] = NO_ACTIVITY_INFO) -> None:
        """
        Add a new peer to this manager.
        :param peer_pk: The public key of the peer to add.
        :param round_active: The round that we last heard from this peer.
        """
        if peer_pk in self.last_active:
            return

        self.logger.info("Participant %s adding participant %s to local view",
                         self.get_my_short_id(), self.get_short_id(peer_pk))
        self.last_active[peer_pk] = round_active

    def remove_peer(self, peer_pk) -> None:
        """
        Remove this peer from this manager.
        :param peer_pk: The public key of the peer to remove.
        """
        self.last_active.pop(peer_pk, None)

    def peer_is_in_view(self, peer_pk: bytes) -> bool:
        return peer_pk in self.last_active

    def get_active_peers(self) -> List[bytes]:
        return [peer_pk for peer_pk, last_round_active in self.last_active.items() if last_round_active != WENT_OFFLINE]

    def get_num_peers(self) -> int:
        """
        Return the number of peers in the local view.
        """
        return len(self.get_active_peers())

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

    def flush_last_active_pending(self) -> None:
        """
        Flush the pending changes to the population view.
        """
        if self.last_active_pending:
            self.logger.info("Participant %s flushing pending changes to population view", self.get_my_short_id())
            self.update_last_active(self.last_active_pending)
            self.last_active_pending = {}

    def update_last_active(self, other_last_active: Dict[bytes, int]) -> None:
        """
        Reconcile the differences between two population views.
        """
        for peer_pk, last_round_active in other_last_active.items():
            if last_round_active == WENT_OFFLINE:
                self.last_active[peer_pk] = WENT_OFFLINE
            else:
                if self.peer_is_in_view(peer_pk) and self.last_active[peer_pk] != WENT_OFFLINE:
                    self.last_active[peer_pk] = max(self.last_active[peer_pk], other_last_active[peer_pk])
                else:
                    # This seems to be a new node joining
                    self.logger.info("Participant %s adding newly joined peer %s to local view",
                                     self.get_my_short_id(), self.get_short_id(peer_pk))
                    self.last_active[peer_pk] = other_last_active[peer_pk]
