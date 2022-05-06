from asyncio import Future, sleep, ensure_future
from binascii import hexlify

import pytest

from accdfl.core.community import DFLCommunity, TransmissionMethod
from accdfl.core import NodeMembershipChange

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


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
    DATASET = "mnist"
    MODEL = "linear"
    TRANSMISSION_METHOD = TransmissionMethod.EVA

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch_size = 1

        self.initialize(DFLCommunity, self.NUM_NODES)

        self.experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": 0.1,
            "momentum": 0.0,
            "batch_size": self.batch_size,
            "participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "all_participants": [hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "sample_size": self.SAMPLE_SIZE,
            "num_aggregators": self.NUM_AGGREGATORS,
            "success_fraction": self.SUCCESS_FRACTION,
            "aggregation_timeout": 0.5,
            "ping_timeout": 0.5,
            "inactivity_threshold": 30,

            # These parameters are not available in a deployed environment - only for experimental purposes.
            "target_participants": self.TARGET_NUM_NODES,
            "samples_per_class": self.SAMPLES_PER_CLASS,
            "local_classes": self.LOCAL_CLASSES,
            "nodes_per_class": self.NODES_PER_CLASS,
            "dataset": self.DATASET,
            "model": self.MODEL,
            "data_distribution": "iid",
        }
        for node in self.nodes:
            node.overlay.train_in_subprocess = False
            node.overlay.setup(self.experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)

    def wait_for_round_completed(self, node, round):
        round_completed_deferred = Future()

        async def on_round_complete(round_nr):
            if round_nr >= round and not round_completed_deferred.done():
                round_completed_deferred.set_result(None)

        node.overlay.round_complete_callback = on_round_complete
        return round_completed_deferred

    def wait_for_num_nodes_in_all_views(self, target_num_nodes):
        test_complete_deferred = Future()

        async def on_round_complete(round_nr):
            if all([node.overlay.peer_manager.get_num_peers(round_nr) == target_num_nodes for node in self.nodes]):
                if not test_complete_deferred.done():
                    test_complete_deferred.set_result(None)

        for node in self.nodes:
            node.overlay.round_complete_callback = on_round_complete

        return test_complete_deferred


class TestDFLCommunityOneNode(TestDFLCommunityBase):
    NUM_NODES = 1
    TARGET_NUM_NODES = NUM_NODES
    SAMPLE_SIZE = NUM_NODES
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10

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
        new_node.overlay.advertise_membership(NodeMembershipChange.JOIN)

        # Perform some rounds so the membership has propagated
        self.nodes[0].overlay.start()
        await self.wait_for_round_completed(self.nodes[0], 2)

        for node in self.nodes:
            assert node.overlay.peer_manager.get_num_peers() == 2
        assert new_node.my_peer.public_key.key_to_bin() in self.nodes[0].overlay.peer_manager.last_active


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
        for node in self.nodes:
            node.overlay.is_active = True

        aggregator, other_node = self.nodes[0], self.nodes[1]
        model = aggregator.overlay.model_manager.model

        await sleep(0.1)

        aggregator.overlay.received_trained_model(aggregator.overlay.my_peer, 1, model)
        aggregator.overlay.received_trained_model(other_node.overlay.my_peer, 1, model)
        await sleep(0.3)
        assert 1 not in aggregator.overlay.aggregating_in_rounds
        assert aggregator.overlay.last_aggregate_round_completed >= 1
        assert 1 not in aggregator.overlay.model_manager.incoming_trained_models

    @pytest.mark.timeout(5)
    async def test_not_start_round_again(self):
        """
        Test whether we are not starting a round that we have already completed.
        """
        aggregator, other_node = self.nodes[0], self.nodes[1]
        model = aggregator.overlay.model_manager.model
        other_node.overlay.last_round_completed = 2
        other_node.overlay.received_aggregated_model(aggregator.overlay.my_peer, 1, model)
        assert not other_node.overlay.is_pending_task_active("round_2")

    @pytest.mark.timeout(5)
    async def test_not_start_aggregate_again(self):
        """
        Test whether we are not starting aggregation for a round that we have already completed.
        """
        aggregator, other_node = self.nodes[0], self.nodes[1]
        model = other_node.overlay.model_manager.model
        aggregator.overlay.last_aggregate_round_completed = 1
        aggregator.overlay.received_trained_model(other_node.overlay.my_peer, 1, model)
        assert not aggregator.overlay.is_pending_task_active("aggregate_1")

    @pytest.mark.timeout(5)
    async def test_ping_succeed(self):
        """
        Test pinging a single peer.
        """
        self.nodes[0].overlay.is_active = True
        self.nodes[1].overlay.is_active = True
        res = await self.nodes[0].overlay.ping_peer(1234, self.nodes[1].overlay.my_peer.public_key.key_to_bin(), 1)
        assert res[2]

    @pytest.mark.timeout(5)
    async def test_ping_fail(self):
        """
        Test pinging a single peer.
        """
        self.nodes[0].overlay.is_active = True
        res = await self.nodes[0].overlay.ping_peer(1234, self.nodes[1].overlay.my_peer.public_key.key_to_bin(), 1)
        assert not res[2]


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


class TestDFLCommunityFiveNodesOneJoining(TestDFLCommunityBase):
    NUM_NODES = 5
    TARGET_NUM_NODES = 6
    SAMPLE_SIZE = 1
    NODES_PER_CLASS = [TARGET_NUM_NODES] * 10

    @pytest.mark.timeout(10)
    async def test_new_node_joining(self):
        new_node = self.create_node()
        self.add_node_to_experiment(new_node)
        await self.introduce_nodes()

        # Start all nodes
        for ind in range(self.NUM_NODES):
            self.nodes[ind].overlay.start()

        self.experiment_data["participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        self.experiment_data["all_participants"].append(hexlify(new_node.my_peer.public_key.key_to_bin()).decode())
        new_node.overlay.setup(self.experiment_data, None, transmission_method=self.TRANSMISSION_METHOD)
        new_node.overlay.advertise_membership(NodeMembershipChange.JOIN)
        self.nodes[-1].overlay.start()

        await self.wait_for_num_nodes_in_all_views(self.TARGET_NUM_NODES)


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

        await self.wait_for_num_nodes_in_all_views(self.NUM_NODES - 1)

    @pytest.mark.timeout(5)
    async def test_node_crashing(self):
        """
        Test whether a node that leaves gracefully will eventually be removed from the population views' of others.
        """

        # Start all nodes
        for ind in range(self.NUM_NODES):
            self.nodes[ind].overlay.start()

        await sleep(0.1)  # Progress the network
        self.nodes[0].overlay.go_offline(2)

        await self.wait_for_num_nodes_in_all_views(self.NUM_NODES - 1)
