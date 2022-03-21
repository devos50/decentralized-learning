from asyncio import sleep

from accdfl.core.community import DFLCommunity

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDFLCommunity(TestBase):
    NUM_NODES = 2

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch_size = 1

        experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": self.batch_size,
        }

        self.initialize(DFLCommunity, self.NUM_NODES, working_directory=":memory:")
        for node in self.nodes:
            node.overlay.setup(experiment_data)

    async def test_audit(self):
        await self.nodes[0].overlay.train()
        assert len(self.nodes[0].overlay.data_store.data_items) == self.batch_size
        assert len(self.nodes[0].overlay.model_store.models) == 2
        await self.deliver_messages()
        assert len(self.nodes[1].overlay.persistence.get_all_blocks())
        assert await self.nodes[1].overlay.audit(self.nodes[0].my_peer.public_key.key_to_bin(), 1)
        assert len(self.nodes[1].overlay.data_store.data_items) == self.batch_size
