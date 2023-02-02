import os
from asyncio import get_event_loop
from binascii import hexlify

from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import LearningSettings, SessionSettings, DLSettings
from ipv8.configuration import ConfigBuilder
from simulations.dl import ExponentialTwoGraph, GetDynamicOnePeerSendRecvRanks
from simulations.settings import SimulationSettings, DLAccuracyMethod

from simulations.learning_simulation import LearningSimulation


class DLSimulation(LearningSimulation):

    def __init__(self, settings: SimulationSettings) -> None:
        super().__init__(settings)
        self.num_round_completed = 0

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.settings.bypass_model_transfers:
            builder.add_overlay("DLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("DLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.settings.learning_rate,
            momentum=self.settings.momentum,
            batch_size=self.settings.batch_size
        )

        dl_settings = DLSettings(topology=self.settings.topology or "ring")

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.settings.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            dl=dl_settings,
            model=self.settings.model,
            alpha=self.settings.alpha,
            partitioner=self.settings.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.settings.train_device_name,
        )

        self.model_manager = ModelManager(None, self.session_settings, 0)

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, i=ind: self.on_round_complete(i, round_nr)
            node.overlays[0].setup(self.session_settings)

        self.build_topology()

        if self.settings.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

    def build_topology(self):
        if self.session_settings.dl.topology == "ring":
            # Build a simple ring topology
            print("Building a ring topology")
            for ind, node in enumerate(self.nodes):
                nb_node = self.nodes[(ind + 1) % len(self.nodes)]
                node.overlays[0].neighbours = [nb_node.overlays[0].my_peer.public_key.key_to_bin()]
        elif self.session_settings.dl.topology == "exp-one-peer":
            G = ExponentialTwoGraph(len(self.nodes))
            for node_ind in range(len(self.nodes)):
                g = GetDynamicOnePeerSendRecvRanks(G, node_ind)
                nb_ids = [next(g)[0][0] for _ in range(len(list(G.neighbors(node_ind))) - 1)]
                for nb_ind in nb_ids:
                    nb_pk = self.nodes[nb_ind].overlays[0].my_peer.public_key.key_to_bin()
                    self.nodes[node_ind].overlays[0].neighbours.append(nb_pk)
        else:
            raise RuntimeError("Unknown DL topology %s" % self.session_settings.dl.topology)

    async def on_round_complete(self, peer_ind: int, round_nr: int):
        self.num_round_completed += 1
        peer_pk = self.nodes[peer_ind].overlays[0].my_peer.public_key.key_to_bin()
        self.model_manager.incoming_trained_models[peer_pk] = self.nodes[peer_ind].overlays[0].model_manager.model
        if self.num_round_completed < len(self.nodes):
            return

        # Everyone completed their round - wrap up
        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        print("All peers completed round %d - bytes up: %d, bytes down: %d" % (round_nr, tot_up, tot_down))
        self.num_round_completed = 0

        # Compute model accuracy
        if round_nr % self.settings.accuracy_logging_interval == 0:
            print("Will compute accuracy for round %d!" % round_nr)
            try:
                if self.settings.dl_accuracy_method == DLAccuracyMethod.AGGREGATE_THEN_TEST:
                    avg_model = self.model_manager.aggregate_trained_models()
                    accuracy, loss = self.evaluator.evaluate_accuracy(avg_model)
                    with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                        out_file.write("%s,DL,%f,%d,%d,%f,%f\n" % (self.settings.dataset, get_event_loop().time(), 0,
                                                                   round_nr, accuracy, loss))
                elif self.settings.dl_accuracy_method == DLAccuracyMethod.TEST_INDIVIDUAL_MODELS:
                    if self.settings.dl_test_mode == "das_jobs":
                        results = self.test_models_with_das_jobs()
                    else:
                        results = self.test_models()

                    for ind, acc_res in results.items():
                        accuracy, loss = acc_res
                        with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                            out_file.write("%s,DL,%f,%d,%d,%f,%f\n" %
                                           (self.settings.dataset, get_event_loop().time(), ind, round_nr, accuracy,
                                            loss))

            except ValueError as e:
                print("Encountered error during evaluation check - dumping models and stopping")
                self.checkpoint_models(round_nr)
                raise e

        # Checkpoint models
        if self.settings.checkpoint_interval and round_nr % self.settings.checkpoint_interval == 0:
            self.checkpoint_models(round_nr)

        self.model_manager.reset_incoming_trained_models()

        for node in self.nodes:
            node.overlays[0].start_next_round()
