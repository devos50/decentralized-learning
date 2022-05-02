import hashlib
from typing import List, Dict, Tuple

from accdfl.core import NodeMembershipChange
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
        self.sample_cache: Dict[Tuple[int, bytes], List[bytes]] = {}

    def get_sample_for_round(self, round: int, exclude_peer: bytes = None, custom_view: Dict = None) -> List[bytes]:
        if not custom_view and (round, exclude_peer) in self.sample_cache:
           return self.sample_cache[(round, exclude_peer)]

        hashes = []
        if custom_view:
            population_view = [peer_pk for peer_pk, info in custom_view.items() if info[1][1] != NodeMembershipChange.LEAVE]
        else:
            population_view = self.peer_manager.get_active_peers()
        population_view = sorted(population_view)

        for peer_id in population_view:
            if peer_id == exclude_peer:
                continue
            h = hashlib.md5(b"%s-%d" % (peer_id, round))
            hashes.append((peer_id, h.digest()))
        hashes = sorted(hashes, key=lambda t: t[1])
        sample = [t[0] for t in hashes[:self.sample_size]]

        if not custom_view:
            self.sample_cache[(round, exclude_peer)] = sample

        return sample

    def get_aggregators_for_round(self, round: int, custom_view: Dict = None) -> List[bytes]:
        derived_sample = self.get_sample_for_round(round, custom_view=custom_view)
        return derived_sample[:self.num_aggregators]

    def is_participant_in_round(self, peer_id: bytes, round: int, custom_view: Dict = None) -> bool:
        return peer_id in self.get_sample_for_round(round, custom_view=custom_view)

    def is_aggregator_in_round(self, peer_id: bytes, round: int, custom_view: Dict = None) -> bool:
        return peer_id in self.get_aggregators_for_round(round, custom_view=custom_view)
