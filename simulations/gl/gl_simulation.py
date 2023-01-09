import os
from asyncio import get_event_loop
from binascii import hexlify
from typing import Optional, List

from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import LearningSettings, SessionSettings, GLSettings
from ipv8.configuration import ConfigBuilder
from simulations.settings import SimulationSettings

from simulations.learning_simulation import LearningSimulation


class GLSimulation(LearningSimulation):

    def __init__(self, settings: SimulationSettings) -> None:
        super().__init__(settings)
        self.model_manager: Optional[ModelManager] = None

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("GLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.settings.learning_rate,
            momentum=self.settings.momentum,
            batch_size=self.settings.batch_size
        )

        gl_settings = GLSettings(self.settings.gl_round_timeout)

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.settings.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            gl=gl_settings,
            data_distribution=self.settings.data_distribution,
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
        """
        Build a fully connected topology where all peers know each other.
        """
        for node in self.nodes:
            for nb_node in self.nodes:
                if node == nb_node:
                    continue
                node.overlays[0].neighbours.append(nb_node.overlays[0].my_peer.public_key.key_to_bin())

    async def on_round_complete(self, peer_ind: int, round_nr: int):
        # Compute model accuracy
        if round_nr % self.settings.accuracy_logging_interval == 0:
            print("Will compute accuracy of peer %d for round %d!" % (peer_ind, round_nr))
            try:
                print("Testing model of peer %d on device %s..." % (peer_ind + 1, self.settings.accuracy_device_name))
                model = self.nodes[peer_ind].overlays[0].model_manager.model
                accuracy, loss = self.evaluator.evaluate_accuracy(
                    model, device_name=self.settings.accuracy_device_name)
                with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,GL,%f,%d,%d,%f,%f\n" %
                                   (self.settings.dataset, get_event_loop().time(),
                                    peer_ind, round_nr, accuracy, loss))
            except ValueError as e:
                print("Encountered error during evaluation check - dumping all models and stopping")
                self.checkpoint_models(round_nr)
                raise e

        # Checkpoint the model
        if self.settings.checkpoint_interval and round_nr % self.settings.checkpoint_interval == 0:
            self.checkpoint_model(peer_ind, round_nr)