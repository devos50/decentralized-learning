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
            client_momentum=self.args.client_momentum,
            server_momentum=self.args.server_momentum,
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

        tot_up, tot_down = self.get_bw_totals()
        train_time: float = self.get_total_train_time()

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
                out_file.write("%s,%d,%g,%d,%s,%f,%d,%d,%f,%f,%d,%d,%f\n" % (self.args.dataset, self.args.seed, self.args.client_learning_rate, self.args.local_steps, "diloco", cur_time,
                                                                 ind, round_nr, accuracy, loss, tot_up, tot_down, train_time))

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
        await super().start_simulation()
