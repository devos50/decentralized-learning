import os
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from typing import List

import torch

from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import LearningSettings, SessionSettings, DLSettings
from ipv8.configuration import ConfigBuilder
from simulations.dl import ExponentialTwoGraph, GetDynamicOnePeerSendRecvRanks

from simulations.learning_simulation import LearningSimulation


class DLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.num_round_completed = 0
        self.participants_ids: List[int] = []
        self.best_accuracy: float = 0.0

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.args.bypass_model_transfers:
            builder.add_overlay("DLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("DLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        if self.args.active_participants:
            self.logger.info("Initial active participants: %s", self.args.active_participants)
            start_ind, end_ind = self.args.active_participants.split("-")
            start_ind, end_ind = int(start_ind), int(end_ind)
            participants_pks = [hexlify(self.nodes[ind].overlays[0].my_peer.public_key.key_to_bin()).decode()
                            for ind in range(start_ind, end_ind)]
            self.participants_ids = list(range(start_ind, end_ind))
        else:
            participants_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes]
            self.participants_ids = list(range(len(self.nodes)))

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        dl_settings = DLSettings(topology=self.args.topology or "ring")

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=participants_pks,
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            dataset_base_path=self.args.dataset_base_path,
            dl=dl_settings,
            model=self.args.model,
            alpha=self.args.alpha,
            partitioner=self.args.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.args.train_device_name,
            bypass_training=self.args.bypass_training,
        )

        self.model_manager = ModelManager(None, self.session_settings, 0)

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, i=ind: self.on_round_complete(i, round_nr)
            node.overlays[0].setup(self.session_settings)

        self.build_topology()

        if self.args.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

    def build_topology(self):
        self.logger.info("Building a %s topology", self.session_settings.dl.topology)
        if self.session_settings.dl.topology == "ring":
            # Build a simple ring topology
            for ind in self.participants_ids:
                nb_node = self.nodes[(ind + 1) % len(self.participants_ids)]
                self.nodes[ind].overlays[0].neighbours = [nb_node.overlays[0].my_peer.public_key.key_to_bin()]
        elif self.session_settings.dl.topology == "exp-one-peer":
            G = ExponentialTwoGraph(len(self.participants_ids))
            for node_ind in range(len(self.participants_ids)):
                g = GetDynamicOnePeerSendRecvRanks(G, node_ind)
                nb_ids = [next(g)[0][0] for _ in range(len(list(G.neighbors(node_ind))) - 1)]
                for nb_ind in nb_ids:
                    nb_pk = self.nodes[self.participants_ids[0] + nb_ind].overlays[0].my_peer.public_key.key_to_bin()
                    self.nodes[self.participants_ids[0] + node_ind].overlays[0].neighbours.append(nb_pk)
        else:
            raise RuntimeError("Unknown DL topology %s" % self.session_settings.dl.topology)

    def test_avg_model(self, round_nr: int, write_results: bool = True):
        avg_model = self.model_manager.aggregate_trained_models()
        accuracy, loss = self.evaluator.evaluate_accuracy(avg_model)
        self.logger.info("Accuracy of central model for round %d: %f (loss: %f)", round_nr, accuracy, loss)
        if write_results:
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,DL,%f,%d,%d,%f,%f\n" % (self.args.dataset, get_event_loop().time(), 0,
                                                           round_nr, accuracy, loss))

        if self.args.store_best_models and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(avg_model.state_dict(), os.path.join(self.data_dir, "best.model"))

    async def on_round_complete(self, peer_ind: int, round_nr: int):
        self.num_round_completed += 1
        peer_pk = self.nodes[peer_ind].overlays[0].my_peer.public_key.key_to_bin()
        self.model_manager.incoming_trained_models[peer_pk] = self.nodes[peer_ind].overlays[0].model_manager.model
        if self.num_round_completed < len(self.session_settings.participants):
            return

        # Everyone completed their round - wrap up
        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        self.logger.info("All peers completed round %d - bytes up: %d, bytes down: %d" % (round_nr, tot_up, tot_down))
        self.num_round_completed = 0

        # Compute model accuracy
        if round_nr % self.args.accuracy_logging_interval == 0:
            self.logger.info("Will compute accuracy for round %d!" % round_nr)
            try:
                if self.args.dl_accuracy_method == "aggregate":
                    self.test_avg_model(round_nr)
                elif self.args.dl_accuracy_method == "individual":
                    if self.args.dl_test_mode == "das_jobs":
                        results = self.test_models_with_das_jobs()
                    else:
                        results = self.test_models()

                    for ind, acc_res in results.items():
                        accuracy, loss = acc_res
                        with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                            out_file.write("%s,DL,%f,%d,%d,%f,%f\n" %
                                           (self.args.dataset, get_event_loop().time(), ind, round_nr, accuracy,
                                            loss))

                    # Also test the avg. model
                    self.test_avg_model(round_nr, write_results=False)

            except ValueError as e:
                print("Encountered error during evaluation check - dumping models and stopping")
                self.checkpoint_models(round_nr)
                raise e

        # Checkpoint models
        if self.args.checkpoint_interval and round_nr % self.args.checkpoint_interval == 0:
            self.checkpoint_models(round_nr)

        self.model_manager.reset_incoming_trained_models()

        if self.args.rounds and round_nr >= self.args.rounds:
            self.on_simulation_finished()
            self.loop.stop()

        for ind in self.participants_ids:
            self.nodes[ind].overlays[0].start_next_round()
