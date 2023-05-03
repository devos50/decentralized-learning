import os
import random
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from math import floor, log
from typing import List

from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import LearningSettings, SessionSettings, DLSettings

from ipv8.configuration import ConfigBuilder

from simulations.dl import ExponentialTwoGraph, GetDynamicOnePeerSendRecvRanks
from simulations.learning_simulation import LearningSimulation

import networkx as nx


class DLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.num_round_completed = 0
        self.participants_ids: List[int] = []
        self.round_nr: int = 1
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d_dl" % (self.args.peers, self.args.dataset, self.args.seed))

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
            node.overlays[0].setup(self.session_settings)

        self.build_topology()

        if self.args.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

        # Generated the statistics files
        with open(os.path.join(self.data_dir, "round_durations.csv"), "w") as out_file:
            out_file.write("round,duration\n")

    async def start_simulation(self) -> None:
        self.round_start_time = get_event_loop().time()
        for node in self.nodes:
            node.overlays[0].start_round(self.round_nr)
        self.register_task("round_done", self.on_round_done, interval=120)
        self.register_task("check_accuracy", self.compute_all_accuracies, interval=self.args.accuracy_logging_interval)
        await super().start_simulation()

    def on_round_done(self):
        # Check if all nodes have wrapped up
        for node in self.nodes:
            assert len(node.overlays[0].bw_scheduler.incoming_transfers) == 0
            assert len(node.overlays[0].bw_scheduler.outgoing_transfers) == 0

        for node in self.nodes:
            node.overlays[0].aggregate_models()

        self.round_nr += 1
        nodes_started = 0

        for node in self.nodes:
            if node.overlays[0].is_active:
                node.overlays[0].start_round(self.round_nr)
                nodes_started += 1

        print("Round %d started (with %d nodes)" % (self.round_nr, nodes_started))

    def compute_all_accuracies(self):
        cur_time = get_event_loop().time()

        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        self.logger.warning("Computing accuracies for all models, current time: %f, bytes up: %d, bytes down: %d",
                            cur_time, tot_up, tot_down)

        # Put all the models in the model manager
        eligible_nodes = []
        for ind, node in enumerate(self.nodes):
            if not self.nodes[ind].overlays[0].is_active:
                continue

            eligible_nodes.append((ind, node))

        # Don't test all models for efficiency reasons, just up to 20% of the entire network
        eligible_nodes = random.sample(eligible_nodes, min(len(eligible_nodes), int(len(self.nodes) * 0.2)))
        print("Will test accuracy of %d nodes..." % len(eligible_nodes))

        for ind, node in eligible_nodes:
            model = self.nodes[ind].overlays[0].model_manager.model
            self.model_manager.process_incoming_trained_model(b"%d" % ind, model)

        if self.args.dl_accuracy_method == "aggregate":
            if not self.args.bypass_training:
                avg_model = self.model_manager.aggregate_trained_models()
                accuracy, loss = self.evaluator.evaluate_accuracy(avg_model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,DL,%f,%d,%d,%f,%f\n" % (self.args.dataset, get_event_loop().time(), 0,
                                                           int(cur_time), accuracy, loss))
        elif self.args.dl_accuracy_method == "individual":
            # Compute the accuracies of all individual models
            if self.args.dl_test_mode == "das_jobs":
                results = self.test_models_with_das_jobs()
            else:
                results = self.test_models()

            for ind, acc_res in results.items():
                accuracy, loss = acc_res
                round_nr = self.nodes[ind].overlays[0].round
                with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,DL,%f,%d,%d,%f,%f\n" %
                                   (self.args.dataset, cur_time, ind, round_nr, accuracy, loss))

        self.model_manager.reset_incoming_trained_models()

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
        elif self.session_settings.dl.topology == "k-regular":
            k: int = floor(log(len(self.nodes), 2))
            self.logger.info("Building %d-regular graph topology", k)
            G = nx.random_regular_graph(k, len(self.nodes))
            for node_ind in range(len(self.nodes)):
                for nb_node_ind in list(G.neighbors(node_ind)):
                    nb_pk = self.nodes[nb_node_ind].overlays[0].my_peer.public_key.key_to_bin()
                    self.nodes[node_ind].overlays[0].neighbours.append(nb_pk)
        else:
            raise RuntimeError("Unknown DL topology %s" % self.session_settings.dl.topology)
