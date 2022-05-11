from asyncio import Future, sleep, ensure_future
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod
from accdfl.core import NodeMembershipChange
from accdfl.core.model_manager import ModelManager

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class FakeModelManager(ModelManager):
    """
    A model manager that does not actually train the model but simply sleeps.
    """
    train_time = 0.001

    async def train(self, in_subprocess: bool = True):
        await sleep(self.train_time)


class TestDFLCommunityBase(TestBase):
    NUM_NODES = 2
    TARGET_NUM_NODES = NUM_NODES
    SAMPLE_SIZE = NUM_NODES
    NUM_AGGREGATORS = 1
    SUCCESS_FRACTION = 1
    LOCAL_CLASSES = 10
    TOTAL_SAMPLES_PER_CLASS = 6
    SAMPLES_PER_CLASS = [TOTAL_SAMPLES_PER_CLASS] * 10
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10
    DATASET = "cifar10"
    TRANSMISSION_METHOD = TransmissionMethod.EVA
    INACTIVITY_THRESHOLD = 10

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch_size = 1

        self.initialize(DFLCommunity, self.NUM_NODES)

        self.experiment_data = {
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": self.batch_size,
            "participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "all_participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "sample_size": self.SAMPLE_SIZE,
            "num_aggregators": self.NUM_AGGREGATORS,
            "success_fraction": self.SUCCESS_FRACTION,
            "aggregation_timeout": 0.5,
            "ping_timeout": 0.1,
            "inactivity_threshold": self.INACTIVITY_THRESHOLD,

            # These parameters are not available in a deployed environment - only for experimental purposes.
            "target_participants": self.TARGET_NUM_NODES,
            "samples_per_class": self.SAMPLES_PER_CLASS,
            "local_classes": self.LOCAL_CLASSES,
            "nodes_per_class": self.NODES_PER_CLASS,
            "dataset": self.DATASET,
            "data_distribution": "iid",
        }
        for node in self.nodes:
            #node.overlay.train_in_subprocess = False
            node.overlay.setup(self.experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)
            # cur_model_mgr = node.overlay.model_manager
            # node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.experiment_data,
            #                                               cur_model_mgr.participant_index)

    def wait_for_round_completed(self, node, round):
        round_completed_deferred = Future()

        async def on_round_complete(round_nr):
            if round_nr >= round and not round_completed_deferred.done():
                round_completed_deferred.set_result(None)

        node.overlay.round_complete_callback = on_round_complete
        return round_completed_deferred

    def wait_for_num_nodes_in_all_views(self, target_num_nodes, exclude_node=None):
        test_complete_deferred = Future()

        async def on_round_complete(round_nr):
            nodes_to_check = [n for n in self.nodes if n != exclude_node]
            if all([node.overlay.peer_manager.get_num_peers(round_nr) == target_num_nodes for node in nodes_to_check]):
                if not test_complete_deferred.done():
                    test_complete_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.round_complete_callback = on_round_complete

        return test_complete_deferred


class TestDFLCommunityOneNode(TestDFLCommunityBase):
    NUM_NODES = 1
    TARGET_NUM_NODES = 100
    SAMPLE_SIZE = NUM_NODES
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10

    @pytest.mark.timeout(50)
    async def test_single_round(self):
        assert self.nodes[0].overlay.did_setup
        self.nodes[0].overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 1)

    @pytest.mark.timeout(5)
    async def test_multiple_rounds(self):
        assert self.nodes[0].overlay.did_setup
        self.nodes[0].overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 5)

    @pytest.mark.timeout(5)
    async def test_aggregate_complete_callback(self):
        test_future = Future()

        async def on_aggregate_complete(round, model):
            assert round == 3  # If we receive a trained model with sample index 4, we're aggregating for round 3.
            assert model
            test_future.set_result(None)

        self.nodes[0].overlay.aggregate_sample_estimate = 2
        self.nodes[0].overlay.aggregate_complete_callback = on_aggregate_complete
        model = self.nodes[0].overlay.model_manager.model
        ensure_future(self.nodes[0].overlay.received_trained_model(self.nodes[0].overlay.my_peer, 4, model))
        await test_future


class TestDFLCommunityOneNodeOneJoining(TestDFLCommunityBase):
    NUM_NODES = 1
    TARGET_NUM_NODES = 2
    SAMPLE_SIZE = NUM_NODES
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10

    @pytest.mark.timeout(5)
    async def test_new_node_joining(self):
        new_node = self.create_node()
        self.add_node_to_experiment(new_node)
        await self.introduce_nodes()

        self.experiment_data["participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        self.experiment_data["all_participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        new_node.overlay.setup(self.experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)
        cur_model_mgr = new_node.overlay.model_manager
        new_node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.experiment_data,
                                                          cur_model_mgr.participant_index)
        new_node.overlay.advertise_membership(NodeMembershipChange.JOIN)

        # Perform some rounds so the membership has propagated
        self.nodes[0].overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 2)

        for node in self.nodes:
            assert node.overlay.peer_manager.get_num_peers() == 2
        assert new_node.my_peer.public_key.key_to_bin() in self.nodes[0].overlay.peer_manager.last_active


class TestDFLCommunityTwoNodes(TestDFLCommunityBase):

    @pytest.mark.timeout(5)
    async def test_eva_send_model(self):
        """
        Test an EVA model transfer between two nodes.
        """
        for node in self.nodes:
            node.overlay.is_active = True

        model = self.nodes[1].overlay.model_manager.model
        res = await self.nodes[0].overlay.eva_send_model(1, model, "aggregated_model", {}, self.nodes[1].overlay.my_peer)
        assert res

    @pytest.mark.timeout(5)
    async def test_start_train_on_aggregated_model(self):
        """
        Test whether we are starting training when receiving an aggregated model.
        """
        self.nodes[0].overlay.train_sample_estimate = 1
        model = self.nodes[1].overlay.model_manager.model
        self.nodes[0].overlay.received_aggregated_model(self.nodes[0].overlay.my_peer, 2, model)
        assert self.nodes[0].overlay.ongoing_training_task_name

    @pytest.mark.timeout(5)
    async def test_not_start_train_on_stale_model(self):
        """
        Test whether we are not starting training when receiving an old aggregated model.
        """
        self.nodes[0].overlay.train_sample_estimate = 5
        model = self.nodes[1].overlay.model_manager.model
        self.nodes[0].overlay.received_aggregated_model(self.nodes[0].overlay.my_peer, 2, model)
        assert not self.nodes[0].overlay.ongoing_training_task_name

    @pytest.mark.timeout(5)
    async def test_interrupt_train_on_newer_aggregated_model(self):
        """
        Test whether we interrupt model training and start to training the newer model when receiving an aggregated model.
        """
        self.nodes[0].overlay.train_sample_estimate = 2
        self.nodes[0].overlay.model_manager.train_time = 0.1
        model = self.nodes[1].overlay.model_manager.model
        self.nodes[0].overlay.received_aggregated_model(self.nodes[0].overlay.my_peer, 2, model)
        assert self.nodes[0].overlay.ongoing_training_task_name == "round_2"

        # New incoming model
        self.nodes[0].overlay.received_aggregated_model(self.nodes[0].overlay.my_peer, 3, model)
        assert not self.nodes[0].overlay.is_pending_task_active("round_2")
        assert self.nodes[0].overlay.ongoing_training_task_name == "round_3"

    @pytest.mark.timeout(5)
    async def test_start_aggregated_on_trained_model(self):
        """
        Test whether we start aggregating when receiving a trained model.
        """
        self.nodes[0].overlay.aggregate_sample_estimate = 1
        self.nodes[0].overlay.model_manager.parameters["sample_size"] = 2
        model = self.nodes[1].overlay.model_manager.model
        await self.nodes[0].overlay.received_trained_model(self.nodes[0].overlay.my_peer, 2, model)
        assert self.nodes[0].overlay.aggregate_start_time
        assert len(self.nodes[0].overlay.model_manager.incoming_trained_models) == 1

    @pytest.mark.timeout(5)
    async def test_not_start_aggregated_on_stale_trained_model(self):
        """
        Test whether we do not start aggregating when receiving an older trained model.
        """
        self.nodes[0].overlay.aggregate_sample_estimate = 5
        model = self.nodes[1].overlay.model_manager.model
        await self.nodes[0].overlay.received_trained_model(self.nodes[0].overlay.my_peer, 4, model)
        assert not self.nodes[0].overlay.aggregate_start_time

    @pytest.mark.timeout(5)
    async def test_reset_aggregate_on_newer_trained_model(self):
        """
        Test whether we reset an ongoing aggregation when receiving a newer trained model.
        """
        self.nodes[0].overlay.aggregate_sample_estimate = 1
        self.nodes[0].overlay.model_manager.parameters["sample_size"] = 2
        model = self.nodes[1].overlay.model_manager.model
        await self.nodes[0].overlay.received_trained_model(self.nodes[0].overlay.my_peer, 2, model)
        assert self.nodes[0].overlay.aggregate_start_time
        assert len(self.nodes[0].overlay.model_manager.incoming_trained_models) == 1

        # Another incoming trained model for a later round - this should reset the current aggregation
        await self.nodes[0].overlay.received_trained_model(self.nodes[1].overlay.my_peer, 3, model)
        assert len(self.nodes[0].overlay.model_manager.incoming_trained_models) == 1

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
    async def test_ping_succeed(self):
        """
        Test pinging a single peer.
        """
        self.nodes[0].overlay.is_active = True
        self.nodes[1].overlay.is_active = True
        res = await self.nodes[0].overlay.ping_peer(1234, self.nodes[1].overlay.my_peer.public_key.key_to_bin())
        assert res[1]

    @pytest.mark.timeout(5)
    async def test_ping_fail(self):
        """
        Test pinging a single peer.
        """
        self.nodes[0].overlay.is_active = True
        res = await self.nodes[0].overlay.ping_peer(1234, self.nodes[1].overlay.my_peer.public_key.key_to_bin())
        assert not res[1]


class TestDFLCommunityFiveNodes(TestDFLCommunityBase):
    NUM_NODES = 5
    TARGET_NUM_NODES = NUM_NODES
    SAMPLE_SIZE = 3
    NUM_AGGREGATORS = 2
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10
    TOTAL_SAMPLES_PER_CLASS = 10

    @pytest.mark.timeout(5)
    async def test_single_round(self):
        for node in self.nodes:
            node.overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 1)

    @pytest.mark.timeout(10)
    async def test_many_rounds(self):
        for node in self.nodes:
            node.overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 50)

    @pytest.mark.timeout(5)
    async def test_get_available_peers(self):
        """
        Test getting available participants in a sample.
        """
        for node in self.nodes:
            node.overlay.is_active = True

        available_peers = await self.nodes[0].overlay.determine_available_peers_for_sample(1, 1)
        assert len(available_peers) == 1

        available_peers = await self.nodes[0].overlay.determine_available_peers_for_sample(1, 5)
        assert len(available_peers) == 5

        # Make two nodes unavailable
        self.nodes[0].overlay.is_active = False
        self.nodes[1].overlay.is_active = False
        available_peers = await self.nodes[2].overlay.determine_available_peers_for_sample(1, 3)
        assert self.nodes[0].overlay.my_peer.public_key.key_to_bin() not in available_peers
        assert self.nodes[1].overlay.my_peer.public_key.key_to_bin() not in available_peers


class TestDFLCommunityFiveNodesOneJoining(TestDFLCommunityBase):
    NUM_NODES = 5
    TARGET_NUM_NODES = 6
    SAMPLE_SIZE = 1
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10
    INACTIVITY_THRESHOLD = 100

    @pytest.mark.timeout(10)
    async def test_new_node_joining(self):
        new_node = self.create_node()
        self.add_node_to_experiment(new_node)
        await self.introduce_nodes()
        new_node.overlay.train_in_subprocess = False

        # Start all nodes
        for ind in range(self.NUM_NODES):
            self.nodes[ind].overlay.start()

        self.experiment_data["participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        self.experiment_data["all_participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        new_node.overlay.setup(self.experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)
        cur_model_mgr = new_node.overlay.model_manager
        new_node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.experiment_data,
                                                          cur_model_mgr.participant_index)
        new_node.overlay.advertise_membership(NodeMembershipChange.JOIN)
        self.nodes[-1].overlay.start(advertise_join=True)

        await self.wait_for_num_nodes_in_all_views(self.TARGET_NUM_NODES, exclude_node=self.nodes[-1])

        # Check whether this has been recorded in the population view history
        for node in self.nodes:
            if node == self.nodes[-1]:
                continue
            assert len(node.overlay.active_peers_history) == 2


class TestDFLCommunityFiveNodesOneLeaving(TestDFLCommunityBase):
    NUM_NODES = 5
    TARGET_NUM_NODES = NUM_NODES
    SAMPLE_SIZE = 2
    NUM_AGGREGATORS = 2
    SUCCESS_FRACTION = 0.5
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10

    @pytest.mark.timeout(5)
    async def test_node_leaving(self):
        """
        Test whether a node that leaves gracefully will eventually be removed from the population views' of others.
        """
        for ind in range(self.NUM_NODES):
            self.nodes[ind].overlay.start()

        await sleep(0.1)  # Progress the network
        self.nodes[0].overlay.go_offline()

        await self.wait_for_num_nodes_in_all_views(self.NUM_NODES - 1, exclude_node=self.nodes[0])

        node_pk = self.nodes[0].overlay.my_peer.public_key.key_to_bin()
        for node in self.nodes[1:]:
            assert node.overlay.peer_manager.last_active[node_pk][1][0] == 1

    @pytest.mark.timeout(10)
    async def test_node_crashing(self):
        """
        Test whether a node that leaves gracefully will eventually be removed from the population views' of others.
        """

        # Start all nodes
        for ind in range(self.NUM_NODES):
            self.nodes[ind].overlay.start()

        await sleep(0.2)  # Progress the network
        self.nodes[0].overlay.go_offline(graceful=False)

        await self.wait_for_num_nodes_in_all_views(self.NUM_NODES - 1, exclude_node=self.nodes[0])
