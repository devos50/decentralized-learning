import os
import random
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from typing import Dict, List, Optional

import torch

from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.model_manager import ModelManager
from accdfl.core.models import create_base_model
from accdfl.core.session_settings import DiLoCoSettings, LearningSettings, SessionSettings

from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation


class DiLoCoSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.num_round_completed = 0
        self.latest_accuracy_check_round: int = 0
        self.last_round_complete_time: Optional[float] = None
        self.participants_ids: List[int] = []
        self.round_nr: int = 1
        self.round_completed_counts: Dict[int, int] = {}
        self.round_durations: List[float] = []
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d_diloco" % (self.args.peers, self.args.dataset, self.args.seed))
        self.nodes_done_in_round: int = 0

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("DiLoCoCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()
        participants_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes]
        self.participants_ids = list(range(len(self.nodes)))

        # Setup the training process
        learning_settings = LearningSettings(
            client_learning_rate=self.args.client_learning_rate,
            client_optimizer=self.args.client_optimizer,
            server_learning_rate=self.args.server_learning_rate,
            server_optimizer=self.args.server_optimizer,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        diloco_settings = DiLoCoSettings()

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=participants_pks,
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            dataset_base_path=self.args.dataset_base_path,
            diloco=diloco_settings,
            model=self.args.model,
            alpha=self.args.alpha,
            partitioner=self.args.partitioner,
            eva_block_size=1000,
            bypass_training=self.args.bypass_training,
            device=self.device,
        )

        split_datasets = self.create_datasets()

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, model, i=ind: self.on_round_complete(i, round_nr, model)
            node.overlays[0].setup(self.session_settings, self.dataset)
            node.overlays[0].model_manager.model_trainer.setup_dataset(split_datasets[ind], self.tokenizer, self.data_collator)

        # Inject the nodes and ourselves in each community
        for ind, node in enumerate(self.nodes):
            node.overlays[0].simulation = self
            node.overlays[0].nodes = self.nodes
            node.overlays[0].node_id = ind

        if not self.args.bypass_training:
            self.evaluator = ModelEvaluator(self.session_settings)
            self.evaluator.setup_dataset(self.test_dataset, self.tokenizer, self.data_collator)

        # Generated the statistics files
        with open(os.path.join(self.data_dir, "round_durations.csv"), "w") as out_file:
            out_file.write("round,duration\n")

    async def on_round_complete(self, ind: int, round_nr: int, model):
        if round_nr not in self.round_completed_counts:
            self.round_completed_counts[round_nr] = 0
        self.round_completed_counts[round_nr] += 1
        if self.round_completed_counts[round_nr] < len(self.session_settings.participants):
            return
        
        self.round_completed_counts.pop(round_nr)

        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        cur_time = get_event_loop().time()
        print("Round %d completed @ t=%f - bytes up: %d, bytes down: %d" % (round_nr, cur_time, tot_up, tot_down))

        if round_nr > self.latest_accuracy_check_round:
            if not self.last_round_complete_time:
                self.round_durations.append(cur_time)
            else:
                self.round_durations.append(cur_time - self.last_round_complete_time)
            self.last_round_complete_time = cur_time

        if self.args.accuracy_logging_interval > 0 and round_nr % self.args.accuracy_logging_interval == 0 and \
                round_nr > self.latest_accuracy_check_round:

            print("Will compute accuracy for round %d!" % round_nr)
            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(model)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                group = "\"s=%d, a=%d\"" % (self.args.sample_size, self.args.num_aggregators)
                out_file.write("%s,%d,%g,%s,%f,%d,%d,%f,%f\n" % (self.args.dataset, self.args.seed, self.args.client_learning_rate, group, get_event_loop().time(),
                                                                 ind, round_nr, accuracy, loss))

            self.latest_accuracy_check_round = round_nr

        if self.args.rounds and round_nr >= self.args.rounds:
            self.on_simulation_finished()
            self.loop.stop()

        # Otherwise, start the next round
        self.round_nr += 1
        for node in self.nodes:
            node.overlays[0].start_round(self.round_nr)

    async def start_simulation(self) -> None:
        self.round_start_time = get_event_loop().time()
        for node in self.nodes:
            node.overlays[0].start_round(self.round_nr)
        if self.args.accuracy_logging_interval_is_in_sec:
            self.register_task("check_accuracy", self.compute_all_accuracies, interval=self.args.accuracy_logging_interval)
        await super().start_simulation()

    def on_node_round_done(self):
        self.nodes_done_in_round += 1
        if self.nodes_done_in_round == len(self.nodes):
            self.on_round_done()

    def on_round_done(self):
        self.logger.error("Round %d done", self.round_nr)
        transfers_to_kill = 0
        for node in self.nodes:
            if node.overlays[0].bw_scheduler.outgoing_transfers:
                for ongoing_transfer in node.overlays[0].bw_scheduler.outgoing_transfers:
                    self.logger.warning("Transfer %s still going on after round completed", ongoing_transfer)
                    transfers_to_kill += 1

            node.overlays[0].bw_scheduler.kill_all_transfers()

        if transfers_to_kill > 0:
            self.logger.error("Killed %d transfers", transfers_to_kill)

        # Should we check the accuracy?
        if not self.args.accuracy_logging_interval_is_in_sec and self.args.accuracy_logging_interval > 0 and self.round_nr % self.args.accuracy_logging_interval == 0:
            self.compute_all_accuracies()

        if self.args.rounds and self.round_nr >= self.args.rounds:
            self.on_simulation_finished()
            self.loop.stop()

        self.round_nr += 1
        self.nodes_done_in_round = 0
        nodes_started = 0

        for node in self.nodes:
            if node.overlays[0].is_active:
                node.overlays[0].start_round(self.round_nr)
                nodes_started += 1

        self.logger.error("Round %d started (with %d nodes)", self.round_nr, nodes_started)

    def compute_all_accuracies(self):
        cur_time = get_event_loop().time()
        tot_up, tot_down = self.get_bw_totals()
        train_time: float = self.get_total_train_time()

        self.logger.warning("Computing accuracies for all models, current time: %f, bytes up: %d, bytes down: %d, total train time: %f",
                            cur_time, tot_up, tot_down, train_time)

        # Put all the models in the model manager
        eligible_nodes = []
        for ind, node in enumerate(self.nodes):
            if not self.nodes[ind].overlays[0].is_active:
                continue

            eligible_nodes.append((ind, node))

        # Don't test all models for efficiency reasons, just up to 100% of the entire network
        FRACTION = 1.0
        eligible_nodes = random.sample(eligible_nodes, min(len(eligible_nodes), int(len(self.nodes) * FRACTION)))
        print("Will test accuracy of %d nodes..." % len(eligible_nodes))

        self.model_manager = ModelManager(self.peft_model, self.session_settings, 0)
        self.model_manager.aggregator = self.nodes[0].overlays[0].aggregator
        self.model_manager.global_adapter = self.nodes[0].overlays[0].model_manager.global_adapter

        for ind, node in eligible_nodes:
            adapter = self.nodes[ind].overlays[0].model_manager.adapter
            self.model_manager.process_incoming_trained_adapter(b"%d" % ind, adapter)

        if self.args.dl_accuracy_method == "aggregate":
            if not self.args.bypass_training:
                self.model_manager.aggregate_trained_adapters()
                accuracy, loss = self.evaluator.evaluate_accuracy(self.peft_model, adapter_to_test="global")
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,%d,%g,%d,%s,%f,%d,%d,%f,%f,%d,%d,%f\n" % (self.args.dataset, self.args.seed, self.args.learning_rate, self.args.local_steps, "DL" if not self.args.el else "EL",
                                                                 cur_time, 0, self.round_nr, accuracy, loss, tot_up, tot_down, train_time))
        elif self.args.dl_accuracy_method == "individual":
            results = self.test_models()

            for ind, acc_res in results.items():
                accuracy, loss = acc_res
                round_nr = self.nodes[ind].overlays[0].round
                with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,%d,%g,%d,%s,%f,%d,%d,%f,%f,%d,%d,%f\n" %
                                   (self.args.dataset, self.args.seed, self.args.learning_rate, self.args.local_steps, "DL" if not self.args.el else "EL",
                                    cur_time, ind, round_nr, accuracy, loss, tot_up, tot_down, train_time))

        self.model_manager.reset_incoming_trained_adapters()
