from asyncio import gather, Future
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod

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

        self.initialize(DFLCommunity, self.NUM_NODES)

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

    def test_train(self):
        """
        Test one model train step by one node.
        """
        self.nodes[0].overlay.train()
