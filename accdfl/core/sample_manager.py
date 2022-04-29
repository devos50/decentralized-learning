import hashlib
from typing import List

from accdfl.core.peer_manager import PeerManager


class SampleManager:
    """
    The SampleManager class is responsible for deriving samples and determine who the aggregators in a particular
    sample are.
    """

    def __init__(self, peer_manager: PeerManager, sample_size: int, num_aggregators: int):
        self.peer_manager: PeerManager = peer_manager
        self.sample_size = sample_size
        self.num_aggregators = num_aggregators

    def get_sample_for_round(self, round: int, exclude_peer: bytes = None) -> List[bytes]:
        hashes = []
        for peer_id in self.peer_manager.peers:
            if peer_id == exclude_peer:
                continue
            h = hashlib.md5(b"%s-%d" % (peer_id, round))
            hashes.append((peer_id, h.digest()))
        hashes = sorted(hashes, key=lambda t: t[1])
        return [t[0] for t in hashes[:self.sample_size]]

    def get_aggregators_for_round(self, round: int) -> List[bytes]:
        derived_sample = self.get_sample_for_round(round)
        return derived_sample[:self.num_aggregators]

    def is_participant_in_round(self, peer_id: bytes, round: int) -> bool:
        return peer_id in self.get_sample_for_round(round)

    def is_aggregator_in_round(self, peer_id: bytes, round: int) -> bool:
        return peer_id in self.get_aggregators_for_round(round)
