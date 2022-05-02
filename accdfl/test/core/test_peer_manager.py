import pytest

from accdfl.core import NodeMembershipChange
from accdfl.core.peer_manager import PeerManager, NO_ACTIVITY_INFO


@pytest.fixture
def peer_manager():
    return PeerManager(b"test")


def test_add_peer(peer_manager):
    peer_manager.add_peer(b"test2")
    assert peer_manager.get_active_peers()
    assert b"test2" in peer_manager.last_active
    assert peer_manager.last_active[b"test2"] == (NO_ACTIVITY_INFO, (0, NodeMembershipChange.JOIN))

    # Try adding the peer again
    peer_manager.add_peer(b"test2")
    assert peer_manager.get_num_peers() == 1

    # Add a peer with a particular round active
    peer_manager.add_peer(b"test3", round_active=3)
    assert peer_manager.last_active[b"test3"] == (3, (0, NodeMembershipChange.JOIN))


def test_remove_peer(peer_manager):
    peer_manager.add_peer(b"test2")
    assert peer_manager.get_num_peers() == 1
    peer_manager.remove_peer(b"test2")
    assert peer_manager.get_num_peers() == 0
    assert b"test2" not in peer_manager.last_active


def test_update_peer_activity(peer_manager):
    peer_manager.add_peer(b"test2")
    peer_manager.update_peer_activity(b"test2", 3)
    assert peer_manager.last_active[b"test2"] == (3, (0, NodeMembershipChange.JOIN))

    # Older message should be ignored
    peer_manager.update_peer_activity(b"test2", 2)
    assert peer_manager.last_active[b"test2"] == (3, (0, NodeMembershipChange.JOIN))


def test_update_last_active(peer_manager):
    peer_manager.add_peer(b"test2")
    other_last_active = {b"test2": (2, (1, NodeMembershipChange.JOIN))}
    peer_manager.update_last_active(other_last_active)
    assert peer_manager.last_active[b"test2"][0] == 2

    # This should be ignored
    other_last_active = {b"test2": (1, (1, NodeMembershipChange.JOIN))}
    peer_manager.update_last_active(other_last_active)
    assert peer_manager.last_active[b"test2"][0] == 2

    other_last_active = {b"test2": (2, (3, NodeMembershipChange.LEAVE))}
    peer_manager.update_last_active(other_last_active)
    assert peer_manager.last_active[b"test2"][1][0] == 3

    # This should be ignored
    other_last_active = {b"test2": (2, (2, NodeMembershipChange.JOIN))}
    peer_manager.update_last_active(other_last_active)
    assert peer_manager.last_active[b"test2"][1][0] == 3


def test_get_my_short_id(peer_manager):
    assert peer_manager.get_my_short_id()


def test_get_short_id(peer_manager):
    assert peer_manager.get_short_id(b"a" * 64)
    assert peer_manager.get_short_id(b"test") == peer_manager.get_my_short_id()
