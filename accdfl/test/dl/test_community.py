from asyncio import sleep
from binascii import hexlify

import pytest

from accdfl.core.session_settings import LearningSettings, SessionSettings, DLSettings
from accdfl.dl.community import DLCommunity
from accdfl.test.util.fake_model_manager import FakeModelManager

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestDLCommunityBase(TestBase):
    NUM_NODES = 2
    TARGET_NUM_NODES = NUM_NODES
    DATASET = "cifar10"

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()

        self.initialize(DLCommunity, self.NUM_NODES)

        learning_settings = LearningSettings(
            learning_rate=0.1,
            momentum=0.0,
            batch_size=1
        )

        dl_settings = DLSettings()

        self.settings = SessionSettings(
            dataset=self.DATASET,
            work_dir=self.temporary_directory(),
            learning=learning_settings,
            participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=self.TARGET_NUM_NODES,
            dl=dl_settings,
            train_in_subprocess=False,
        )

        for ind, node in enumerate(self.nodes):
            node.overlay.setup(self.settings)
            cur_model_mgr = node.overlay.model_manager
            node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.settings,
                                                          cur_model_mgr.participant_index)

            # Build a simple ring topology
            nb_node = self.nodes[(ind + 1) % len(self.nodes)]
            node.overlay.neighbours = [nb_node.overlay.my_peer.public_key.key_to_bin()]


class TestDLCommunity(TestDLCommunityBase):

    @pytest.mark.asyncio
    async def test_two_nodes_round(self):
        await self.introduce_nodes()
        for node in self.nodes:
            assert node.overlay.did_setup
            node.overlay.start()

        await sleep(0.2)

        assert all([node.overlay.round == 2 for node in self.nodes])
