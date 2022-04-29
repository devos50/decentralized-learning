import pytest

from accdfl.core.peer_manager import PeerManager, NO_ACTIVITY_INFO


@pytest.fixture
def peer_manager():
    return PeerManager(b"test")


def test_add_peer(peer_manager):
    peer_manager.add_peer(b"test2")
    assert peer_manager.peers
    assert b"test2" in peer_manager.last_active
    assert peer_manager.last_active[b"test2"] == NO_ACTIVITY_INFO

    # Try adding the peer again
    peer_manager.add_peer(b"test2")
    assert len(peer_manager.peers) == 1

    # Add a peer with a particular round active
    peer_manager.add_peer(b"test3", round_active=3)
    assert peer_manager.last_active[b"test3"] == 3


def test_remove_peer(peer_manager):
    peer_manager.add_peer(b"test2")
    assert len(peer_manager.peers) == 1
    peer_manager.remove_peer(b"test2")
    assert len(peer_manager.peers) == 0
    assert b"test2" not in peer_manager.last_active


def test_update_last_activity(peer_manager):
    peer_manager.add_peer(b"test2")
    peer_manager.update_peer_activity(b"test2", 3)
    assert peer_manager.last_active[b"test2"] == 3

    # Older message should be ignored
    peer_manager.update_peer_activity(b"test2", 2)
    assert peer_manager.last_active[b"test2"] == 3


def test_get_my_short_id(peer_manager):
    assert peer_manager.get_my_short_id()


def test_get_short_id(peer_manager):
    assert peer_manager.get_short_id(b"a" * 64)
    assert peer_manager.get_short_id(b"test") == peer_manager.get_my_short_id()
