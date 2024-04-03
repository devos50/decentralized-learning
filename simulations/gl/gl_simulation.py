import os
import random
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from typing import Dict, List

import torch

from accdfl.core.model_manager import ModelManager
from accdfl.core.models import create_model
from accdfl.core.session_settings import LearningSettings, SessionSettings, GLSettings
from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation


class GLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.cohorts: Dict[int, List[int]] = {}
        self.node_to_cohort: Dict[int, int] = {}
        self.min_val_loss_per_cohort: Dict[int, float] = {}

        if self.args.cohort_file is not None:
            # Read the cohort organisations
            with open(os.path.join("data", self.args.cohort_file)) as cohort_file:
                for line in cohort_file.readlines():
                    parts = line.strip().split(",")
                    self.cohorts[int(parts[0])] = [int(n) for n in parts[1].split("-")]
                    self.min_val_loss_per_cohort[int(parts[0])] = 1000000

            # Create the node -> cohort mapping
            for cohort_ind, nodes_in_cohort in self.cohorts.items():
                for node_ind in nodes_in_cohort:
                    self.node_to_cohort[node_ind] = cohort_ind

        partitioner_str = self.args.partitioner if self.args.partitioner != "dirichlet" else "dirichlet%g" % self.args.alpha
        datadir_name = "n_%d_%s_%s_sd%d_gl" % (self.args.peers, self.args.dataset, partitioner_str, self.args.seed)
        if self.cohorts:
            datadir_name += "_ct%d_p%d" % (len(self.cohorts), self.args.cohort_participation)

        self.data_dir = os.path.join("data", datadir_name)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.args.bypass_model_transfers:
            builder.add_overlay("GLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("GLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        gl_settings = GLSettings(self.args.gl_round_timeout)

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            dataset_base_path=self.args.dataset_base_path,
            validation_set_fraction=self.args.validation_set_fraction,
            compute_validation_loss_global_model=self.args.compute_validation_loss_global_model,
            compute_validation_loss_updated_model=self.args.compute_validation_loss_updated_model,
            gl=gl_settings,
            partitioner=self.args.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.args.train_device_name,
            bypass_training=self.args.bypass_training,
        )

        self.model_manager = ModelManager(None, self.session_settings, 0)

        for ind, node in enumerate(self.nodes):
            node.overlays[0].setup(self.session_settings)

            # Initialize the models of each node. We have to do this here because GL is first sharing, then training so the
            # model would not be initialized first.
            torch.manual_seed(self.session_settings.model_seed)
            node.overlays[0].model_manager.model = create_model(self.session_settings.dataset, architecture=self.session_settings.model)

        self.build_topology()

        if self.args.accuracy_logging_interval > 0:
            interval = self.args.accuracy_logging_interval
            self.logger.info("Registering logging interval task that triggers every %d seconds", interval)
            self.register_task("acc_check", self.on_accuracy_check_interval, delay=interval, interval=interval)

        with open(os.path.join(self.data_dir, "losses.csv"), "w") as out_file:
            out_file.write("cohorts,seed,alpha,participation,cohort,peer,type,time,round,loss\n")

    def on_accuracy_check_interval(self):
        self.compute_all_accuracies()
        self.register_validation_losses()
        for cohort in self.cohorts.keys():
            self.save_aggregated_model_of_cohort(cohort)

    def register_validation_losses(self):
        cur_time = get_event_loop().time()
        with open(os.path.join(self.data_dir, "losses.csv"), "a") as out_file:
            for node_ind, node in enumerate(self.nodes):
                cohort_ind = self.node_to_cohort[node_ind]
                trainer = node.overlays[0].model_manager.model_trainer
                for round_nr, train_loss in trainer.training_losses.items():
                    out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (
                    len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation,
                    cohort_ind, node_ind, "train", int(cur_time), round_nr, train_loss))
                trainer.training_losses = {}

                if self.args.compute_validation_loss_global_model:
                    for round_nr, val_loss in trainer.validation_loss_global_model.items():
                        out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (
                        len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation,
                        cohort_ind, node_ind, "val_global", int(cur_time), round_nr, val_loss))
                    trainer.validation_loss_global_model = {}

                if self.args.compute_validation_loss_updated_model:
                    for round_nr, val_loss in trainer.validation_loss_updated_model.items():
                        out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (
                        len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation,
                        cohort_ind, node_ind, "val_updated", int(cur_time), round_nr, val_loss))
                    trainer.validation_loss_updated_model = {}

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
                out_file.write("%s,GL,%f,%d,%d,%f,%f\n" % (self.args.dataset, get_event_loop().time(), 0,
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
                    out_file.write("%s,GL,%f,%d,%d,%f,%f\n" %
                                   (self.args.dataset, cur_time, ind, round_nr, accuracy, loss))

        self.model_manager.reset_incoming_trained_models()

    def save_aggregated_model_of_cohort(self, cohort: int):
        model_manager: ModelManager = ModelManager(None, self.session_settings, 0)
        for node_ind in self.cohorts[cohort]:
            model = self.nodes[node_ind].overlays[0].model_manager.model.cpu()
            model_manager.process_incoming_trained_model(b"%d" % node_ind, model)

        avg_model = model_manager.aggregate_trained_models()
        models_dir = os.path.join(self.data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        cur_time = get_event_loop().time()
        torch.save(avg_model.state_dict(),
                   os.path.join(models_dir, "c%d_0_%d_0_last.model" % (cohort, cur_time)))

    def build_topology(self):
        """
        Build a k-out topology. This is compatible with the experiment results in the GL papers.
        """
        if self.cohorts:
            for cohort_ind, node_indices_in_cohort in self.cohorts.items():
                nodes_in_cohort = [self.nodes[ind] for ind in node_indices_in_cohort]
                for node in nodes_in_cohort:
                    other_nodes = [n for n in nodes_in_cohort if n != node]
                    node.overlays[0].nodes = other_nodes
        else:
            for node in self.nodes:
                other_nodes = [n for n in self.nodes if n != node]
                node.overlays[0].nodes = other_nodes
