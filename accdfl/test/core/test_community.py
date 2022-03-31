from asyncio import gather, Future, sleep
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod
from accdfl.test.util.network_utils import NetworkUtils
from accdfl.test.util.torrent_http_tracker import TorrentHTTPTracker

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDFLCommunityBase(TestBase):
    NUM_NODES = 2
    NUM_ROUNDS = 2
    LOCAL_CLASSES = 10
    TOTAL_SAMPLES_PER_CLASS = 6
    SAMPLES_PER_CLASS = [TOTAL_SAMPLES_PER_CLASS] * 10
    NODES_PER_CLASS = [NUM_NODES] * 10
    DATASET = "mnist"
    MODEL = "linear"
    TRANSMISSION_METHOD = TransmissionMethod.EVA

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch_size = 1

        self.initialize(DFLCommunity, self.NUM_NODES, working_directory=":memory:")

        experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": self.batch_size,
            "participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "rounds": self.NUM_ROUNDS,
            "sample_size": self.NUM_NODES,

            # These parameters are not available in a deployed environment - only for experimental purposes.
            "samples_per_class": self.SAMPLES_PER_CLASS,
            "local_classes": self.LOCAL_CLASSES,
            "nodes_per_class": self.NODES_PER_CLASS,
            "dataset": self.DATASET,
            "model": self.MODEL,
        }
        for node in self.nodes:
            node.overlay.is_local_test = True
            node.overlay.setup(experiment_data, self.temporary_directory(), transmission_method=self.TRANSMISSION_METHOD)


class TestDFLCommunityTwoNodes(TestDFLCommunityBase):

    async def test_train(self):
        """
        Test one model train step by one node.
        """
        await self.nodes[0].overlay.train()
        assert len(self.nodes[0].overlay.data_store.data_items) == self.batch_size
        assert len(self.nodes[0].overlay.model_store.models) == 2

    # async def test_audit(self):
    #     """
    #     Test the audit procedure.
    #     """
    #     await self.nodes[0].overlay.train()
    #     await self.deliver_messages()
    #     blocks = self.nodes[1].overlay.persistence.get_all_blocks()
    #     assert blocks
    #     block = blocks[0]
    #     assert block.transaction["old_model"] != block.transaction["new_model"]
    #
    #     assert await self.nodes[1].overlay.audit(self.nodes[0].my_peer.public_key.key_to_bin(), 1)
    #     assert len(self.nodes[1].overlay.data_store.data_items) == self.batch_size

    async def test_single_round(self):
        """
        Test whether a single round of training can be completed successfully.
        """
        assert len(self.nodes[0].overlay.get_participants_for_round(1)) == self.NUM_NODES
        assert self.nodes[0].overlay.is_participant_for_round(1)
        await gather(*[node.overlay.participate_in_round() for node in self.nodes])

    async def test_multiple_round(self):
        """
        Test multiple rounds of training.
        """
        round_2_completed = []
        round_2_completed_deferred = Future()

        async def on_round_complete(round_nr, _):
            if round_nr == 2:
                round_2_completed.append(True)
                if len(round_2_completed) == self.NUM_NODES:
                    round_2_completed_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.round_complete_callback = on_round_complete
            node.overlay.start()

        await round_2_completed_deferred

    async def test_multiple_round_smaller_sample(self):
        """
        Test multiple rounds of training with a smaller sample size.
        """
        round_2_completed_deferred = Future()

        async def on_round_complete(round_nr, _):
            if round_nr == 2:
                round_2_completed_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.sample_size = 1
            node.overlay.round_complete_callback = on_round_complete
            node.overlay.start()

        await round_2_completed_deferred

    async def test_compute_accuracy(self):
        """
        Test computing the accuracy of a model.
        """
        await self.nodes[0].overlay.train()
        accuracy, loss = await self.nodes[0].overlay.compute_accuracy()
        assert accuracy > 0

    async def test_epoch(self):
        """
        Test training for an entire epoch.
        """
        steps_in_epoch = int((self.TOTAL_SAMPLES_PER_CLASS * 10) / len(self.nodes) / self.batch_size)
        for _ in range(steps_in_epoch - 1):
            assert not await self.nodes[0].overlay.train()
        assert await self.nodes[0].overlay.train()

    def test_get_iid_dataset_statistics(self):
        stats = self.nodes[0].overlay.dataset.get_statistics()
        assert stats
        assert "total_samples" in stats
        assert stats["total_samples"] == self.TOTAL_SAMPLES_PER_CLASS * 10 / len(self.nodes)
        assert "samples_per_class" in stats
        assert all(n == self.TOTAL_SAMPLES_PER_CLASS / len(self.nodes) for n in stats["samples_per_class"])


class TestDFLCommunityTwoNodesNonIID(TestDFLCommunityBase):
    LOCAL_CLASSES = 5
    NODES_PER_CLASS = [1] * 10

    def test_get_non_iid_dataset_statistics(self):
        stats = self.nodes[0].overlay.dataset.get_statistics()
        assert stats
        assert "total_samples" in stats
        assert sum(stats["samples_per_class"]) == self.TOTAL_SAMPLES_PER_CLASS * 5


class TestDFLCommunityTwoNodesCIFAR10(TestDFLCommunityBase):
    DATASET = "cifar10"
    MODEL = "gnlenet"

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        """
        Test whether a single round of training can be completed successfully.
        """
        assert len(self.nodes[0].overlay.get_participants_for_round(1)) == self.NUM_NODES
        assert self.nodes[0].overlay.is_participant_for_round(1)
        await gather(*[node.overlay.participate_in_round() for node in self.nodes])

    async def test_multiple_rounds(self):
        """
        Test multiple rounds of training.
        """
        round_2_completed = []
        round_2_completed_deferred = Future()

        async def on_round_complete(round_nr, _):
            if round_nr == 2:
                round_2_completed.append(True)
                if len(round_2_completed) == self.NUM_NODES:
                    round_2_completed_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.round_complete_callback = on_round_complete
            node.overlay.start()

        await round_2_completed_deferred


class TestDFLCommunityTwoNodesLibtorrent(TestDFLCommunityBase):
    TRANSMISSION_METHOD = TransmissionMethod.LIBTORRENT

    @pytest.mark.timeout(10)
    async def test_single_round(self):
        """
        Test whether a single round of training can be completed successfully.
        """
        assert len(self.nodes[0].overlay.get_participants_for_round(1)) == self.NUM_NODES
        assert self.nodes[0].overlay.is_participant_for_round(1)
        await gather(*[node.overlay.participate_in_round() for node in self.nodes])

    @pytest.mark.timeout(15)
    async def test_multiple_rounds(self):
        """
        Test multiple rounds of training with the libtorrent transmission engine.
        """
        round_2_completed = []
        round_2_completed_deferred = Future()

        async def on_round_complete(round_nr, _):
            if round_nr == 2:
                round_2_completed.append(True)
                if len(round_2_completed) == self.NUM_NODES:
                    round_2_completed_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.round_complete_callback = on_round_complete
            node.overlay.start()

        await round_2_completed_deferred


class TestDFLCommunityTwoNodesLibtorrentTracker(TestDFLCommunityBase):
    TRANSMISSION_METHOD = TransmissionMethod.LIBTORRENT

    @pytest.mark.timeout(10)
    async def test_single_round_with_http_tracker(self):
        """
        Test whether a single round of training can be completed successfully when using a HTTP tracker.
        """
        tracker_port = NetworkUtils().get_random_free_port()
        http_tracker = TorrentHTTPTracker(tracker_port)
        await http_tracker.start()

        for node in self.nodes:
            node.overlay.torrent_download_manager.trackers.append("http://127.0.0.1:%d/announce" % tracker_port)

        assert len(self.nodes[0].overlay.get_participants_for_round(1)) == self.NUM_NODES
        assert self.nodes[0].overlay.is_participant_for_round(1)
        await gather(*[node.overlay.participate_in_round() for node in self.nodes])

        assert http_tracker.peers

        await http_tracker.stop()
