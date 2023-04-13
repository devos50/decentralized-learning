from asyncio import sleep
from binascii import hexlify

import pytest

from accdfl.core.session_settings import GLSettings, LearningSettings, SessionSettings
from accdfl.gl.community import GLCommunity
from accdfl.test.util.fake_model_manager import FakeModelManager

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestGLCommunityBase(TestBase):
    NUM_NODES = 4
    TARGET_NUM_NODES = NUM_NODES
    DATASET = "cifar10"

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()

        self.initialize(GLCommunity, self.NUM_NODES)

        learning_settings = LearningSettings(
            learning_rate=0.1,
            momentum=0.0,
            batch_size=1,
            weight_decay=0
        )

        gl_settings = GLSettings(round_timeout=0.1)

        self.settings = SessionSettings(
            dataset=self.DATASET,
            work_dir=self.temporary_directory(),
            learning=learning_settings,
            participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=self.TARGET_NUM_NODES,
            gl=gl_settings,
            train_in_subprocess=False,
        )

        for ind, node in enumerate(self.nodes):
            node.overlay.setup(self.settings)
            cur_model_mgr = node.overlay.model_manager
            node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.settings,
                                                          cur_model_mgr.participant_index)

            # Build a fully connected topology
            nb_node = self.nodes[(ind + 1) % len(self.nodes)]
            node.overlay.neighbours = [nb_node.overlay.my_peer.public_key.key_to_bin()]


class TestGLCommunity(TestGLCommunityBase):

    @pytest.mark.asyncio
    async def test_two_nodes_round(self):
        await self.introduce_nodes()
        for node in self.nodes:
            assert node.overlay.did_setup
            node.overlay.start()

        await sleep(0.5)

        # Make sure that nodes have made progress.
        assert all([node.overlay.round > 2 for node in self.nodes])
