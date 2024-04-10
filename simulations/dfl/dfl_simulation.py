import glob
import os
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify
from random import Random
from typing import List, Dict, Optional, Tuple

import torch

from accdfl.core import NodeMembershipChange
from accdfl.core.session_settings import DFLSettings, LearningSettings, SessionSettings
from accdfl.core.peer_manager import PeerManager

from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation
from simulations.logger import SimulationLoggerAdapter


class DFLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.latest_accuracy_check_round: int = 0
        self.last_round_complete_time: Optional[float] = None
        self.current_aggregated_model = None
        self.current_aggregated_model_per_cohort: Dict = {}
        self.current_aggregated_model_round: int = 0
        self.current_aggregated_model_round_per_cohort: Dict = {}
        self.cumsums_per_cohort = {}
        self.rolling_avgs_per_cohort = {}
        self.min_val_loss_per_cohort: Dict[int, float] = {}
        self.round_durations: List[float] = []
        self.best_accuracy: float = 0.0
        self.data_dir = None
        self.cohorts: Dict[int, List[int]] = {}
        self.cohorts_completed = set()
        self.aggregator_per_cohort: Dict[int, int] = {}
        self.sample_size_per_cohort: Dict[int, int] = {}

        if self.args.cohort_file is not None:
            # Read the cohort organisations
            with open(os.path.join("data", self.args.cohort_file)) as cohort_file:
                for line in cohort_file.readlines():
                    parts = line.strip().split(",")
                    self.cohorts[int(parts[0])] = [int(n) for n in parts[1].split("-")]
                    self.min_val_loss_per_cohort[int(parts[0])] = 1000000

        # If we only activate one cohort (specified by the --cohort flag), remove all other cohorts from the equation
        if self.args.cohort is not None:
            self.cohorts = {self.args.cohort: self.cohorts[self.args.cohort]}

        partitioner_str = self.args.partitioner if self.args.partitioner != "dirichlet" else "dirichlet%g" % self.args.alpha
        datadir_name = "n_%d_%s_%s_sd%d" % (
            self.args.peers, self.args.dataset, partitioner_str, self.args.seed)
        if self.cohorts:
            datadir_name += "_ct%d_p%d" % (len(self.cohorts), self.args.cohort_participation)
        if self.args.cohort is not None:
            assert self.args.cohort < len(self.cohorts), "Cohort index (%d) exceeding total num of cohorts (%d)" % (self.args.cohort, len(self.cohorts))
            datadir_name += "_c%d" % self.args.cohort
        datadir_name += "_dfl"
        self.data_dir = os.path.join("data", datadir_name)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.args.bypass_model_transfers:
            builder.add_overlay("DFLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("DFLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        if self.args.active_participants:
            self.logger.info("Initial active participants: %s", self.args.active_participants)
            start_ind, end_ind = self.args.active_participants.split("-")
            start_ind, end_ind = int(start_ind), int(end_ind)
            participants_pks = [hexlify(self.nodes[ind].overlays[0].my_peer.public_key.key_to_bin()).decode()
                            for ind in range(start_ind, end_ind)]
            participants_ids = list(range(start_ind, end_ind))
        elif self.args.cohort_file and self.args.cohort is not None:  # We're just running a single cohort
            participants_ids = self.cohorts[self.args.cohort]
            participants_pks = [hexlify(self.nodes[ind].overlays[0].my_peer.public_key.key_to_bin()).decode()
                                for ind in participants_ids]
        else:
            participants_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes]
            participants_ids = list(range(len(self.nodes)))

        if self.args.sample_size == 0:
            self.args.sample_size = len(participants_ids)
            self.logger.info("Setting sample size to %d" % self.args.sample_size)

        # Determine who will be the aggregator
        aggregator_peer_pk = None
        lowest_latency_peer_id = -1
        if self.args.fix_aggregator:
            if self.args.cohort_file is not None:
                for cohort_ind, cohort_peers in self.cohorts.items():
                    lowest_latency_peer_id = self.determine_peer_with_lowest_median_latency(cohort_peers)
                    self.aggregator_per_cohort[cohort_ind] = lowest_latency_peer_id
            else:
                lowest_latency_peer_id = self.determine_peer_with_lowest_median_latency(participants_ids)
                aggregator_peer_pk = self.nodes[lowest_latency_peer_id].overlays[0].my_peer.public_key.key_to_bin()

        if self.args.cohort_file is not None:
            # Setup cohorts

            # Fix the sample size
            self.args.sample_size = min(len(self.cohorts[0]), self.args.cohort_participation)

            for cohort_ind, cohort_peers in self.cohorts.items():
                aggregator_peer_pk = self.nodes[self.aggregator_per_cohort[cohort_ind]].overlays[0].my_peer.public_key.key_to_bin()
                self.logger.info("Setting up cohort %d with %d peers and aggregator %d...", cohort_ind, len(cohort_peers), self.aggregator_per_cohort[cohort_ind])

                # Set the bandwidth of the aggregating peer to unlimited
                self.nodes[self.aggregator_per_cohort[cohort_ind]].overlays[0].bw_scheduler.bw_limit = -1

                # Fix the sample size
                cohort_sample_size = min(len(cohort_peers), self.args.cohort_participation)
                self.sample_size_per_cohort[cohort_ind] = cohort_sample_size

                learning_settings = LearningSettings(
                    learning_rate=self.args.learning_rate,
                    momentum=self.args.momentum,
                    batch_size=self.args.batch_size,
                    weight_decay=self.args.weight_decay,
                    local_steps=self.args.local_steps,
                )

                dfl_settings = DFLSettings(
                    sample_size=cohort_sample_size,
                    num_aggregators=self.args.num_aggregators,
                    success_fraction=self.args.success_fraction,
                    liveness_success_fraction=self.args.liveness_success_fraction,
                    ping_timeout=5,
                    inactivity_threshold=1000,
                    fixed_aggregator=aggregator_peer_pk,
                    aggregation_timeout=self.args.aggregation_timeout,
                )

                cohort_nodes = [self.nodes[i] for i in cohort_peers]
                cohort_peer_pks = [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in cohort_nodes]
                session_settings = SessionSettings(
                    work_dir=self.data_dir,
                    dataset=self.args.dataset,
                    learning=learning_settings,
                    participants=cohort_peer_pks,
                    all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
                    target_participants=len(self.nodes),
                    dataset_base_path=self.args.dataset_base_path,
                    validation_set_fraction=self.args.validation_set_fraction,
                    compute_validation_loss_global_model=self.args.compute_validation_loss_global_model,
                    compute_validation_loss_updated_model=self.args.compute_validation_loss_updated_model,
                    dfl=dfl_settings,
                    model=self.args.model,
                    alpha=self.args.alpha,
                    partitioner=self.args.partitioner,
                    eva_block_size=1000,
                    is_simulation=True,
                    train_device_name=self.args.train_device_name,
                    bypass_training=self.args.bypass_training,
                    seed=self.args.seed,
                )

                for cohort_peer_ind in cohort_peers:
                    node = self.nodes[cohort_peer_ind]
                    node.overlays[0].aggregate_complete_callback = lambda round_nr, model, train_info, i=cohort_peer_ind: self.on_aggregate_complete(i, round_nr, model, train_info)
                    node.overlays[0].setup(session_settings)
                    node.overlays[0].model_manager.model_trainer.logger = SimulationLoggerAdapter(node.overlays[0].model_manager.model_trainer.logger, {})

                if cohort_ind == 0:
                    self.session_settings = session_settings  # Store one instantiation of these settings for other purposes
        else:
            # Setup the training process
            learning_settings = LearningSettings(
                learning_rate=self.args.learning_rate,
                momentum=self.args.momentum,
                batch_size=self.args.batch_size,
                weight_decay=self.args.weight_decay,
                local_steps=self.args.local_steps,
            )

            dfl_settings = DFLSettings(
                sample_size=self.args.sample_size,
                num_aggregators=self.args.num_aggregators,
                success_fraction=self.args.success_fraction,
                liveness_success_fraction=self.args.liveness_success_fraction,
                ping_timeout=5,
                inactivity_threshold=1000,
                fixed_aggregator=aggregator_peer_pk if self.args.fix_aggregator else None,
                aggregation_timeout=self.args.aggregation_timeout,
            )

            self.session_settings = SessionSettings(
                work_dir=self.data_dir,
                dataset=self.args.dataset,
                learning=learning_settings,
                participants=participants_pks,
                all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
                target_participants=len(self.nodes),
                dataset_base_path=self.args.dataset_base_path,
                validation_set_fraction=self.args.validation_set_fraction,
                compute_validation_loss_global_model=self.args.compute_validation_loss_global_model,
                compute_validation_loss_updated_model=self.args.compute_validation_loss_updated_model,
                dfl=dfl_settings,
                model=self.args.model,
                alpha=self.args.alpha,
                partitioner=self.args.partitioner,
                eva_block_size=1000,
                is_simulation=True,
                train_device_name=self.args.train_device_name,
                bypass_training=self.args.bypass_training,
                seed=self.args.seed,
            )

            for ind, node in enumerate(self.nodes):
                node.overlays[0].aggregate_complete_callback = lambda round_nr, model, train_info, i=ind: self.on_aggregate_complete(i, round_nr, model, train_info)
                node.overlays[0].setup(self.session_settings)
                node.overlays[0].model_manager.model_trainer.logger = SimulationLoggerAdapter(node.overlays[0].model_manager.model_trainer.logger, {})

        # If we fix the aggregator, we assume unlimited upload/download slots
        if self.args.fix_aggregator:
            if self.args.bypass_model_transfers:
                print("Overriding bandwidth limit for peer %d" % lowest_latency_peer_id)
                self.nodes[lowest_latency_peer_id].overlays[0].bw_limit = -1
            else:
                print("Overriding max. EVA transfers for peer %d" % lowest_latency_peer_id)
                self.nodes[lowest_latency_peer_id].overlays[0].eva.settings.max_simultaneous_transfers = 100000

        if self.args.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

        # Generated the statistics files
        with open(os.path.join(self.data_dir, "view_histories.csv"), "w") as out_file:
            out_file.write("peer,update_time,peers\n")

        with open(os.path.join(self.data_dir, "determine_sample_durations.csv"), "w") as out_file:
            out_file.write("peer,start_time,end_time\n")

        with open(os.path.join(self.data_dir, "round_durations.csv"), "w") as out_file:
            out_file.write("time\n")

        with open(os.path.join(self.data_dir, "derived_samples.csv"), "w") as out_file:
            out_file.write("peer,sample_id,sample\n")

        with open(os.path.join(self.data_dir, "losses.csv"), "w") as out_file:
            out_file.write("cohorts,seed,alpha,participation,cohort,peer,type,time,round,loss\n")

        with open(os.path.join(self.data_dir, "events.csv"), "w") as out_file:
            out_file.write("time,peer,round,event\n")

        if self.args.cohort_file:
            with open(os.path.join(self.data_dir, "cohorts_info.csv"), "w") as out_file:
                out_file.write("cohort,time,round,loss,ongoing_cohorts,finished_cohorts\n")

            with open(os.path.join(self.data_dir, "cohorts_data.csv"), "w") as out_file:
                out_file.write("cohort,class,num_samples\n")

        # Start the liveness check (every 5 minutes)
        self.register_task("check_liveness", self.check_liveness, interval=600)

        # Start the model checkpointing process
        if self.args.checkpoint_interval and self.args.checkpoint_interval_is_in_sec:
            self.logger.info("Starting model checkpoint loop every %d sec", self.args.checkpoint_interval)

            if self.args.cohort_file:
                self.register_task("checkpoint", self.checkpoint_cohort_models_interval, interval=self.args.checkpoint_interval)
            else:
                self.register_task("checkpoint", self.checkpoint_model_interval, interval=self.args.checkpoint_interval)

        # Start the model accuracy check process
        if self.args.accuracy_logging_interval and self.args.accuracy_logging_interval_is_in_sec:
            self.logger.info("Starting model accuracy check loop every %d sec", self.args.accuracy_logging_interval)

            if self.args.cohort_file:
                self.register_task("accuracy_check", self.check_cohorts_accuracy_interval, interval=self.args.accuracy_logging_interval)
            else:
                self.register_task("accuracy_check", self.check_accuracy_interval, interval=self.args.accuracy_logging_interval)

    def check_liveness(self):
        # Condition 1: At least one online node is training their model
        one_node_training: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].model_manager.model_trainer.is_training:
                one_node_training = True
                break

        # Condition 2: There is an ongoing model transfer
        one_node_sending: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].bw_scheduler.outgoing_transfers:
                one_node_sending = True
                break

        one_node_aggregating: bool = False
        for node in self.nodes:
            if node.overlays[0].is_active and node.overlays[0].aggregations:
                one_node_aggregating = True
                break

        if not one_node_training and not one_node_sending and not one_node_aggregating:
            self.flush_statistics()
            raise RuntimeError("Liveness violated - MoDeST not making progress anymore")

    def start_nodes_training(self, active_nodes: List) -> None:
        # Update the membership status of inactive peers in all peer managers. This assumption should be
        # reasonable as availability at the very start of the training process can easily be synchronized using an
        # out-of-band mechanism (e.g., published on a website).
        active_nodes_pks = set(node.overlays[0].my_peer.public_key.key_to_bin() for node in active_nodes)
        for node in self.nodes:
            peer_manager: PeerManager = node.overlays[0].peer_manager
            for peer_pk in peer_manager.last_active:
                if peer_pk not in active_nodes_pks:
                    # Toggle the status to inactive as this peer is not active from the beginning
                    peer_info = peer_manager.last_active[peer_pk]
                    peer_manager.last_active[peer_pk] = (peer_info[0], (0, NodeMembershipChange.LEAVE))

        # We will now start round 1. The nodes that participate in the first round are always selected from the pool of
        # active peers. If we use our sampling function, training might not start at all if many offline nodes
        # are selected for the first round.
        rand_sampler = Random(self.args.seed)

        activated_nodes = []

        if self.args.cohort_file:
            # We have to activate some nodes per cohort
            for cohort_ind, cohort_peers in self.cohorts.items():
                sample_size = self.sample_size_per_cohort[cohort_ind]
                peers_to_pick = min(len(cohort_peers), sample_size)
                self.logger.info("Activating %d peers in cohort %d", peers_to_pick, cohort_ind)
                activated_peer_ids = rand_sampler.sample(cohort_peers, peers_to_pick)
                activated_nodes += [self.nodes[peer_id] for peer_id in activated_peer_ids]
        else:
            activated_nodes = rand_sampler.sample(active_nodes, min(len(active_nodes), self.args.sample_size))

        for initial_active_node in activated_nodes:
            overlay = initial_active_node.overlays[0]
            self.logger.info("Activating peer %s in round 1", overlay.peer_manager.get_my_short_id())
            overlay.received_aggregated_model(overlay.my_peer, 1, overlay.model_manager.model)

    def checkpoint_model_interval(self):
        self.logger.info("Checkpointing model...")
        if not self.current_aggregated_model:
            return

        cur_time = get_event_loop().time()
        models_dir = os.path.join(self.data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(self.current_aggregated_model.state_dict(), os.path.join(models_dir, "%d_%d_0.model" % (self.current_aggregated_model_round, cur_time)))

    def checkpoint_cohort_models_interval(self):
        cur_time = get_event_loop().time()
        for cohort_ind in range(len(self.cohorts)):
            self.logger.info("Checkpointing cohort model %d...", cohort_ind)
            if cohort_ind not in self.current_aggregated_model_per_cohort:
                continue

            models_dir = os.path.join(self.data_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(self.current_aggregated_model_per_cohort[cohort_ind].state_dict(), os.path.join(models_dir, "c%d_%d_%d.model" % (cohort_ind, self.current_aggregated_model_round_per_cohort[cohort_ind], cur_time)))

    def check_accuracy_interval(self):
        self.logger.info("Checking accuracy of model...")
        if not self.current_aggregated_model:
            return

        if not self.args.bypass_training:
            accuracy, loss = self.evaluator.evaluate_accuracy(self.current_aggregated_model, device_name=self.args.accuracy_device_name)
        else:
            accuracy, loss = 0, 0

        with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
            bytes_up, bytes_down, train_time, network_time = self.get_aggregated_statistics()
            group = "\"s=%d, a=%d\"" % (self.args.sample_size, self.args.num_aggregators)
            out_file.write("%s,%s,%f,0,%d,%f,%f,%d,%d,%f,%f\n" % (self.args.dataset, group, get_event_loop().time(),
                                                                  self.current_aggregated_model_round, accuracy, loss,
                                                                  bytes_up, bytes_down, train_time, network_time))

    def get_aggregated_statistics_per_cohort(self) -> Dict[int, Tuple[int, int, float, float]]:
        statistics = {}
        for cohort_ind, cohort_peers in self.cohorts.items():
            total_bytes_up: int = 0
            total_bytes_down: int = 0
            total_train_time: float = 0
            total_network_time: float = 0

            for ind in cohort_peers:
                total_bytes_up += self.nodes[ind].overlays[0].endpoint.bytes_up
                total_bytes_down += self.nodes[ind].overlays[0].endpoint.bytes_down
                total_train_time += self.nodes[ind].overlays[0].model_manager.model_trainer.total_training_time
                total_network_time += self.nodes[ind].overlays[0].bw_scheduler.total_time_transmitting

            statistics[cohort_ind] = (total_bytes_up, total_bytes_down, total_train_time, total_network_time)

        return statistics

    def check_cohorts_accuracy_interval(self):
        cohort_statistics = self.get_aggregated_statistics_per_cohort()
        for cohort_ind in range(len(self.cohorts)):
            if cohort_ind not in self.current_aggregated_model_per_cohort:
                continue

            self.logger.info("Checking accuracy of model of cohort %d...", cohort_ind)

            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(self.current_aggregated_model_per_cohort[cohort_ind], device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                bytes_up, bytes_down, train_time, network_time = cohort_statistics[cohort_ind]
                out_file.write("%s,%d,%f,0,%d,%f,%f,%d,%d,%f,%f\n" % (self.args.dataset, cohort_ind, get_event_loop().time(),
                                                                      self.current_aggregated_model_round_per_cohort[cohort_ind], accuracy, loss,
                                                                      bytes_up, bytes_down, train_time, network_time))

    def on_new_validation_loss(self, cohort, loss) -> bool:
        """
        Process a new validation loss. Return True if converged.
        """
        if cohort not in self.cumsums_per_cohort:
            self.cumsums_per_cohort[cohort] = [loss]
        else:
            self.cumsums_per_cohort[cohort].append(self.cumsums_per_cohort[cohort][-1] + loss)

        if cohort not in self.rolling_avgs_per_cohort:
            self.rolling_avgs_per_cohort[cohort] = [loss]
        else:
            list_len = len(self.rolling_avgs_per_cohort[cohort])
            if list_len < self.args.stop_criteria_window_size:
                self.rolling_avgs_per_cohort[cohort].append(self.cumsums_per_cohort[cohort][-1] / len(self.cumsums_per_cohort[cohort]))
            else:
                self.rolling_avgs_per_cohort[cohort].append((self.cumsums_per_cohort[cohort][-1] - self.cumsums_per_cohort[cohort][list_len - self.args.stop_criteria_window_size]) / float(self.args.stop_criteria_window_size))

        # Determine the round where the minimum validation loss stops decreasing
        min_loss = float('inf')
        count_patience = 0
        for avg_loss in self.rolling_avgs_per_cohort[cohort]:
            if avg_loss < min_loss:
                min_loss = avg_loss
                count_patience = 0
            else:
                count_patience += 1

            if count_patience >= self.args.stop_criteria_patience:
                return True
        else:
            return False

    async def on_aggregate_complete(self, ind: int, round_nr: int, model, train_info: Dict[str, float]):
        cohort_training: bool = self.args.cohort_file
        if cohort_training:
            # Which cohort are we in?
            agg_cohort_ind = -1
            for cohort_ind, agg_ind in self.aggregator_per_cohort.items():
                if ind == agg_ind:
                    agg_cohort_ind = cohort_ind
                    break

            if agg_cohort_ind == -1:
                raise RuntimeError("Couldn't associate node %d with any cohort!", ind)

            self.current_aggregated_model_per_cohort[agg_cohort_ind] = model
            self.current_aggregated_model_round_per_cohort[agg_cohort_ind] = round_nr
            self.logger.info("Cohort %d completed round %d", agg_cohort_ind, round_nr)

            # Write away the losses in the model managers of the peers in this cohort
            cur_time = get_event_loop().time()
            with open(os.path.join(self.data_dir, "losses.csv"), "a") as out_file:
                for ind_in_seq, cohort_peer_ind in enumerate(self.cohorts[agg_cohort_ind]):
                    trainer = self.nodes[cohort_peer_ind].overlays[0].model_manager.model_trainer
                    for round_nr, train_loss in trainer.training_losses.items():
                        out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation, agg_cohort_ind, ind_in_seq, "train", int(cur_time), round_nr, train_loss))
                    trainer.training_losses = {}

                    if self.args.compute_validation_loss_global_model:
                        for round_nr, val_loss in trainer.validation_loss_global_model.items():
                            out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation, agg_cohort_ind, ind_in_seq, "val_global", int(cur_time), round_nr, val_loss))
                        trainer.validation_loss_global_model = {}

                    if self.args.compute_validation_loss_updated_model:
                        for round_nr, val_loss in trainer.validation_loss_updated_model.items():
                            out_file.write("%d,%d,%.1f,%d,%d,%d,%s,%d,%d,%f\n" % (len(self.cohorts), self.args.seed, self.args.alpha, self.args.cohort_participation, agg_cohort_ind, ind_in_seq, "val_updated", int(cur_time), round_nr, val_loss))
                        trainer.validation_loss_updated_model = {}

            if self.args.compute_validation_loss_global_model:
                new_avg_loss = train_info["val_loss_global_model"]
                should_stop = self.on_new_validation_loss(agg_cohort_ind, new_avg_loss)

                new_rolling_avg_loss = self.rolling_avgs_per_cohort[agg_cohort_ind][-1]
                self.logger.info("Avg. validation loss of cohort %d: %f, unaveraged: %f", agg_cohort_ind, new_rolling_avg_loss, new_avg_loss)
                if new_rolling_avg_loss < self.min_val_loss_per_cohort[agg_cohort_ind]:
                    self.logger.error("Cohort %d has a lower validation loss: %f - checkpointing model", agg_cohort_ind, new_rolling_avg_loss)
                    self.min_val_loss_per_cohort[agg_cohort_ind] = new_rolling_avg_loss

                    cur_time = get_event_loop().time()
                    models_dir = os.path.join(self.data_dir, "models")
                    os.makedirs(models_dir, exist_ok=True)

                    # Remove old models
                    old_models = glob.glob(os.path.join(models_dir, "c%d_*_best.model" % agg_cohort_ind))
                    for old_model_path in old_models:
                        os.remove(old_model_path)
                    torch.save(model.state_dict(), os.path.join(models_dir, "c%d_%d_%d_0_best.model" % (agg_cohort_ind, round_nr, cur_time)))

                if should_stop:
                    self.logger.error("Validation loss of cohort %d not decreasing - stopping it", agg_cohort_ind)
                    for cohort_peer_ind in self.cohorts[agg_cohort_ind]:
                        self.nodes[cohort_peer_ind].overlays[0].go_offline(graceful=False)
                    self.cohorts_completed.add(agg_cohort_ind)

                    with open(os.path.join(self.data_dir, "cohorts_info.csv"), "a") as out_file:
                        num_cohorts_finished = len(self.cohorts_completed)
                        out_file.write("%d,%f,%d,%f,%d,%d\n" % (agg_cohort_ind, int(get_event_loop().time()), round_nr,
                                                                new_rolling_avg_loss,
                                                                len(self.cohorts) - num_cohorts_finished,
                                                                num_cohorts_finished))

                    # Store the last model
                    cur_time = get_event_loop().time()
                    models_dir = os.path.join(self.data_dir, "models")
                    torch.save(model.state_dict(), os.path.join(models_dir, "c%d_%d_%d_0_last.model" % (agg_cohort_ind, round_nr, cur_time)))

                    # Determine the number of data samples per class
                    dataset = self.nodes[0].overlays[0].model_manager.model_trainer.dataset
                    samples_per_class = [0] * dataset.get_num_classes()
                    for cohort_peer_ind in self.cohorts[agg_cohort_ind]:
                        peer_dataset = self.nodes[cohort_peer_ind].overlays[0].model_manager.model_trainer.dataset
                        for a, (b, clsses) in enumerate(peer_dataset.get_trainset(500, shuffle=False)):
                            for cls in clsses:
                                samples_per_class[cls] += 1

                    with open(os.path.join(self.data_dir, "cohorts_data.csv"), "a") as out_file:
                        for cls_idx, num_samples in enumerate(samples_per_class):
                            out_file.write("%d,%d,%d\n" % (agg_cohort_ind, cls_idx, num_samples))

                    if len(self.cohorts_completed) == len(self.cohorts):
                        exit(0)
        else:
            self.current_aggregated_model = model
            self.current_aggregated_model_round = round_nr

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

        # Checkpoint if needed
        if self.args.checkpoint_interval and not self.args.checkpoint_interval_is_in_sec and round_nr % self.args.checkpoint_interval == 0:
            models_dir = os.path.join(self.data_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(models_dir, "%d_%d_%d.model" % (round_nr, cur_time, ind)))

        # Check accuracy of needed
        if self.args.accuracy_logging_interval > 0 and not self.args.accuracy_logging_interval_is_in_sec and \
                round_nr % self.args.accuracy_logging_interval == 0 and round_nr > self.latest_accuracy_check_round and \
                (not cohort_training or len(self.cohorts) == 1):

            print("Will compute accuracy for round %d!" % round_nr)
            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                bytes_up, bytes_down, train_time, network_time = self.get_aggregated_statistics()
                group = "\"s=%d, a=%d\"" % (self.args.sample_size, self.args.num_aggregators)
                out_file.write("%s,%s,%f,%d,%d,%f,%f,%d,%d,%f,%f\n" % (self.args.dataset, group, get_event_loop().time(),
                                                                       ind, round_nr, accuracy, loss,
                                                                       bytes_up, bytes_down, train_time, network_time))

                if not self.args.bypass_training and self.args.store_best_models and accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    torch.save(model.state_dict(), os.path.join(self.data_dir, "best.model"))

            self.latest_accuracy_check_round = round_nr

        if self.args.rounds and round_nr >= self.args.rounds:
            self.on_simulation_finished()
            self.loop.stop()

    def get_statistics(self) -> Dict:
        statistics = super().get_statistics()

        # Add the BW statistics of individual nodes
        def merge_dicts_sum(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1:
                    if isinstance(value, dict):
                        merge_dicts_sum(dict1[key], value)
                    else:
                        dict1[key] += value
                else:
                    if isinstance(value, dict):
                        dict1[key] = value.copy()
                        merge_dicts_sum(dict1[key], value)
                    else:
                        dict1[key] = value

        agg_bw_in_dict = {}
        agg_bw_out_dict = {}
        for ind, node in enumerate(self.nodes):
            merge_dicts_sum(agg_bw_in_dict, node.overlays[0].bw_in_stats)
            merge_dicts_sum(agg_bw_out_dict, node.overlays[0].bw_out_stats)

        statistics["global"]["bw_in"] = agg_bw_in_dict
        statistics["global"]["bw_out"] = agg_bw_out_dict

        return statistics

    def flush_statistics(self):
        """
        Flush all the statistics generated by nodes.
        """
        super().flush_statistics()

        # Write away the view histories
        with open(os.path.join(self.data_dir, "view_histories.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for update_time, active_peers in node.overlays[0].active_peers_history:
                    if self.args.write_view_histories:
                        active_peers_str = "-".join(active_peers)
                        out_file.write("%d,%f,%s\n" % (peer_id + 1, update_time, active_peers_str))
                node.overlays[0].active_peers_history = []

        # Write the determine sample durations
        with open(os.path.join(self.data_dir, "determine_sample_durations.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for start_time, end_time in node.overlays[0].determine_sample_durations:
                    out_file.write("%d,%f,%f\n" % (peer_id + 1, start_time, end_time))
                node.overlays[0].determine_sample_durations = []

        # Write away the round durations
        with open(os.path.join(self.data_dir, "round_durations.csv"), "a") as out_file:
            for round_duration in self.round_durations:
                out_file.write("%f\n" % round_duration)
            self.round_durations = []

        # Write away the derived samples
        with open(os.path.join(self.data_dir, "derived_samples.csv"), "a") as out_file:
            for peer_id, node in enumerate(self.nodes):
                for sample_id, sample in node.overlays[0].derived_samples:
                    sample_str = "-".join(sorted(sample))
                    out_file.write("%d,%d,%s\n" % (peer_id + 1, sample_id, sample_str))
                node.overlays[0].derived_samples = []

        # Write away all events
        new_events = []
        for node in self.nodes:
            for event in node.overlays[0].events:
                new_events.append(event)
            node.overlays[0].events = []

        if self.args.log_events:
            new_events = sorted(new_events, key=lambda x: x[0])
            with open(os.path.join(self.data_dir, "events.csv"), "a") as out_file:
                for event in new_events:
                    out_file.write("%f,%s,%d,%s\n" % event)
