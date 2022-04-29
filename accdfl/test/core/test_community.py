from asyncio import Future, sleep, ensure_future
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod
from accdfl.core.model import serialize_model

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDFLCommunityBase(TestBase):
    NUM_NODES = 2
    SAMPLE_SIZE = NUM_NODES
    NUM_AGGREGATORS = 1
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
            "sample_size": self.SAMPLE_SIZE,
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
            node.overlay.setup(experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)

    def wait_for_round_completed(self, node, round):
        round_completed_deferred = Future()

        def on_round_complete(round_nr):
            if round_nr >= round and not round_completed_deferred.done():
                round_completed_deferred.set_result(None)

        node.overlay.round_complete_callback = on_round_complete
        return round_completed_deferred


class TestDFLCommunityOneNode(TestDFLCommunityBase):
    NUM_NODES = 1
    SAMPLE_SIZE = NUM_NODES
    NODES_PER_CLASS = [NUM_NODES] * 10

    async def test_start_invalid_round(self):
        with pytest.raises(RuntimeError):
            await self.nodes[0].overlay.participate_in_round(0)

        self.nodes[0].overlay.participating_in_rounds.add(1)
        with pytest.raises(RuntimeError):
            await self.nodes[0].overlay.participate_in_round(1)

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        assert self.nodes[0].overlay.did_setup
        self.nodes[0].overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 1)


class TestDFLCommunityTwoNodes(TestDFLCommunityBase):

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        for node in self.nodes:
            node.overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 1)

    @pytest.mark.timeout(5)
    async def test_multiple_rounds(self):
        for node in self.nodes:
            node.overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 5)

    @pytest.mark.timeout(5)
    async def test_wait_for_aggregated_models(self):
        """
        Test whether the aggregator proceeds when it has received sufficient valid models.
        """
        aggregator, other_node = (self.nodes[0], self.nodes[1]) if self.nodes[0].overlay.my_id in self.nodes[0].overlay.sample_manager.get_aggregators_for_round(2) else (self.nodes[1], self.nodes[0])
        serialized_model = serialize_model(aggregator.overlay.model_manager.model)

        ensure_future(aggregator.overlay.participate_in_round(1))
        await sleep(0.1)

        # Invalid models should be ignored
        await other_node.overlay.received_trained_model(aggregator.overlay.my_peer, 1, serialized_model)
        assert not other_node.overlay.model_manager.incoming_trained_models

        await aggregator.overlay.received_trained_model(aggregator.overlay.my_peer, 1, serialized_model)
        await aggregator.overlay.received_trained_model(other_node.overlay.my_peer, 1, serialized_model)
        await aggregator.overlay.aggregation_deferred
        await sleep(0.1)
        assert 1 not in aggregator.overlay.model_manager.incoming_trained_models


class TestDFLCommunityFiveNodes(TestDFLCommunityBase):
    NUM_NODES = 5
    SAMPLE_SIZE = 3
    NUM_AGGREGATORS = 2
    NODES_PER_CLASS = [NUM_NODES] * 10
    TOTAL_SAMPLES_PER_CLASS = 10

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        for node in self.nodes:
            node.overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 1)
