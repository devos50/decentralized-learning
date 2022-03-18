from accdfl.core.community import DFLCommunity

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDFLCommunity(TestBase):
    NUM_NODES = 2

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()

        experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": 250,
        }

        self.initialize(DFLCommunity, self.NUM_NODES, working_directory=":memory:")
        for node in self.nodes:
            node.overlay.setup(experiment_data)

    async def test_simple(self):
        self.nodes[0].overlay.train()
        self.nodes[1].overlay.audit(self.nodes[0].my_peer.public_key, 1)
