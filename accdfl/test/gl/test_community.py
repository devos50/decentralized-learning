from asyncio import Future, sleep
from binascii import hexlify

import pytest

from accdfl.core.session_settings import LearningSettings, GLSettings, SessionSettings
from accdfl.gl.community import GLCommunity
from accdfl.test.fake_model_manager import FakeModelManager

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestGLCommunityBase(TestBase):
    NUM_NODES = 2
    TARGET_NUM_NODES = NUM_NODES
    DATASET = "cifar10"

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch_size = 1

        self.initialize(GLCommunity, self.NUM_NODES)

        learning_settings = LearningSettings(
            learning_rate=0.1,
            momentum=0.0,
            batch_size=self.batch_size
        )

        gl_settings = GLSettings(
            round_duration=0.1
        )

        self.settings = SessionSettings(
            dataset=self.DATASET,
            work_dir=self.temporary_directory(),
            learning=learning_settings,
            participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=self.TARGET_NUM_NODES,
            data_distribution="iid",
            gl=gl_settings,
            train_in_subprocess=False,
        )

        for node in self.nodes:
            node.overlay.setup(self.settings)
            cur_model_mgr = node.overlay.model_manager
            node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.settings,
                                                          cur_model_mgr.participant_index)

    def wait_for_round_completed(self, node, round):
        round_completed_deferred = Future()

        async def on_round_complete(round_nr):
            if round_nr >= round and not round_completed_deferred.done():
                round_completed_deferred.set_result(None)

        node.overlay.round_complete_callback = on_round_complete
        return round_completed_deferred


class TestGLCommunityOneNode(TestGLCommunityBase):
    NUM_NODES = 2
    TARGET_NUM_NODES = 100
    SAMPLE_SIZE = NUM_NODES

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        assert self.nodes[0].overlay.did_setup
        for node in self.nodes:
            node.overlay.start()
        await sleep(2)
        #await self.wait_for_round_completed(self.nodes[0], 1)
