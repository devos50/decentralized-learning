from asyncio import sleep

from accdfl.core.community import DFLCommunity
from ipv8.test.base import TestBase


class TestDFLCommunity(TestBase):
    NUM_NODES = 2

    def setUp(self):
        super().setUp()
        self.initialize(DFLCommunity, self.NUM_NODES)

    async def test_simple(self):
        with open("../../../my_model.mod", "rb") as in_file:
            dat = in_file.read()
        self.nodes[0].overlay.eva_send_binary(self.nodes[1].my_peer, b"test", dat)
        await sleep(5)
        assert False
