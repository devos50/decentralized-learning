import os
from asyncio import get_event_loop
from binascii import hexlify
from math import log, ceil
from typing import Optional

from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import LearningSettings, SessionSettings, DLSettings
from ipv8.configuration import ConfigBuilder
from simulations.dl import ExponentialTwoGraph, GetDynamicOnePeerSendRecvRanks
from simulations.settings import SimulationSettings

from simulations.learning_simulation import LearningSimulation


class DLSimulation(LearningSimulation):

    def __init__(self, settings: SimulationSettings) -> None:
        super().__init__(settings)
        self.num_round_completed = 0
        self.model_manager: Optional[ModelManager] = None

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
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
            data_distribution=self.settings.data_distribution,
            eva_block_size=1000,
            is_simulation=True,
        )

        self.model_manager = ModelManager(None, self.session_settings, 0)

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, i=ind: self.on_round_complete(i, round_nr)
            node.overlays[0].setup(self.session_settings)

        self.build_topology()

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
        print("All peers completed round %d!" % round_nr)
        self.num_round_completed = 0

        # Compute model accuracy
        if round_nr % self.settings.accuracy_logging_interval == 0:
            avg_model = self.model_manager.average_trained_models()
            print("Will compute accuracy for round %d!" % round_nr)
            accuracy, loss = self.evaluator.evaluate_accuracy(avg_model)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,DL,%f,%d,%d,%f,%f\n" % (self.settings.dataset, get_event_loop().time(), 0,
                                                           round_nr, accuracy, loss))

        self.model_manager.reset_incoming_trained_models()

        for node in self.nodes:
            node.overlays[0].start_next_round()
