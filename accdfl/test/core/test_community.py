from asyncio import sleep, ensure_future, gather

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
            "participants": self.NUM_NODES
        }

        self.initialize(DFLCommunity, self.NUM_NODES, working_directory=":memory:")
        for node in self.nodes:
            node.overlay.setup(experiment_data)

    async def test_train(self):
        """
        Test one model train step by one node.
        """
        await self.nodes[0].overlay.train()
        assert len(self.nodes[0].overlay.data_store.data_items) == self.batch_size
        assert len(self.nodes[0].overlay.model_store.models) == 2

    async def test_audit(self):
        """
        Test the audit procedure.
        """
        await self.nodes[0].overlay.train()
        await self.deliver_messages()
        blocks = self.nodes[1].overlay.persistence.get_all_blocks()
        assert blocks
        block = blocks[0]
        assert block.transaction["old_model"] != block.transaction["new_model"]

        assert await self.nodes[1].overlay.audit(self.nodes[0].my_peer.public_key.key_to_bin(), 1)
        assert len(self.nodes[1].overlay.data_store.data_items) == self.batch_size

    async def test_round(self):
        """
        Test whether a single round of training can be completed successfully.
        """
        await gather(*[node.overlay.advance_round() for node in self.nodes])
        for node in self.nodes:
            assert node.overlay.round == 2
