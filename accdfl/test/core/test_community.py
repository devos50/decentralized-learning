from asyncio import Future
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDFLCommunityBase(TestBase):
    NUM_NODES = 2
    NUM_AGGREGATORS = 1
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

        self.initialize(DFLCommunity, self.NUM_NODES)

        experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": self.batch_size,
            "participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "rounds": self.NUM_ROUNDS,
            "sample_size": self.NUM_NODES,
            "num_aggregators": self.NUM_AGGREGATORS,

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

    async def wait_for_round_completed(self, node, round):
        round_completed_deferred = Future()

        async def on_round_complete(round_nr):
            if round_nr >= round:
                round_completed_deferred.set_result(None)

        node.overlay.round_complete_callback = on_round_complete
        await round_completed_deferred


class TestDFLCommunityTwoNodes(TestDFLCommunityBase):

    async def test_start_invalid_round(self):
        with pytest.raises(RuntimeError):
            await self.nodes[0].overlay.execute_round(0)

        self.nodes[0].overlay.is_participating_in_round = True
        with pytest.raises(RuntimeError):
            await self.nodes[0].overlay.execute_round(1)

        self.nodes[0].overlay.is_participating_in_round = False
        self.nodes[0].overlay.round = 1
        with pytest.raises(RuntimeError):
            await self.nodes[0].overlay.execute_round(1)

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        for node in self.nodes:
            node.overlay.start()

        await self.wait_for_round_completed(self.nodes[0], 1)

    @pytest.mark.timeout(10)
    async def test_wait_for_aggregated_models(self):
        aggregator = self.nodes[0] if self.nodes[0].overlay.my_id in self.nodes[0].overlay.sample_manager.get_aggregators_for_round(1) else self.nodes[1]
        await aggregator.overlay.aggregation_deferred
