import os
import random
import stat
import subprocess
import sys
from asyncio import get_event_loop
from binascii import hexlify
from typing import Optional

import torch

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
        self.model_manager: Optional[ModelManager] = None

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
                        self.test_models_with_das_jobs(round_nr)
                    else:
                        self.test_models(round_nr)

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

    def dump_settings(self):
        """
        Dump the session settings if they do not exist yet.
        """
        settings_file_path = os.path.join(self.session_settings.work_dir, "settings.json")
        if not os.path.exists(settings_file_path):
            with open(settings_file_path, "w") as settings_file:
                settings_file.write(self.session_settings.to_json())

    def test_models_with_das_jobs(self, round_nr: int) -> None:
        """
        Test the accuracy of all models in the model manager by spawning different DAS jobs.
        """
        self.dump_settings()

        # Divide the clients over the DAS nodes
        client_queue = list(range(len(self.model_manager.incoming_trained_models.values())))
        while client_queue:
            self.logger.info("Scheduling new batch on DAS nodes - %d clients left", len(client_queue))

            processes = []
            all_model_ids = set()
            for job_ind in range(self.settings.das_test_subprocess_jobs):
                if not client_queue:
                    continue

                clients_on_this_node = []
                while client_queue and len(clients_on_this_node) < self.settings.das_test_num_models_per_subprocess:
                    client = client_queue.pop(0)
                    clients_on_this_node.append(client)

                # Prepare the input files for the subjobs
                model_ids = []
                for client_id in clients_on_this_node:
                    model_id = random.randint(1, 1000000)
                    while model_id in all_model_ids:
                        model_id = random.randint(1, 1000000)
                    model_ids.append(model_id)
                    all_model_ids.add(model_id)
                    model_file_name = "%d.model" % model_id
                    model_path = os.path.join(self.session_settings.work_dir, model_file_name)
                    node_pk = self.nodes[client_id].overlays[0].my_peer.public_key.key_to_bin()
                    torch.save(self.model_manager.incoming_trained_models[node_pk].state_dict(), model_path)

                import accdfl.util as autil
                script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "evaluate_model.py")

                # Prepare the files and spawn the processes!
                out_file_path = os.path.join(os.getcwd(), "out_%d.log" % job_ind)
                model_ids_str = ",".join(["%d" % model_id for model_id in model_ids])

                train_cmd = "python3 %s %s %s %s" % (script_dir, self.session_settings.work_dir, model_ids_str, self.model_manager.data_dir)
                bash_file_name = "run_%d.sh" % job_ind
                with open(bash_file_name, "w") as bash_file:
                    bash_file.write("""#!/bin/bash
module load cuda11.7/toolkit/11.7
source /home/spandey/venv3/bin/activate
cd %s
export PYTHONPATH=%s
%s
""" % (os.getcwd(), os.getcwd(), train_cmd))
                    st = os.stat(bash_file_name)
                    os.chmod(bash_file_name, st.st_mode | stat.S_IEXEC)

                cmd = "ssh fs3.das6.tudelft.nl \"prun -t 5:00 -np 1 -o %s %s\"" % (out_file_path, os.path.join(os.getcwd(), bash_file_name))
                self.logger.debug("Command: %s", cmd)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append((p, cmd, model_ids))

            for p, cmd, model_ids in processes:
                p.wait()
                self.logger.info("Command %s completed!", cmd)
                if p.returncode != 0:
                    raise RuntimeError("Training subprocess exited with non-zero code %d" % p.returncode)

                # This batch is done! Collect the results...
                for model_id in model_ids:
                    model_file_name = "%d.model" % model_id
                    model_path = os.path.join(self.session_settings.work_dir, model_file_name)
                    os.unlink(model_path)

                    # Read the accuracy and the loss from the file
                    results_file = os.path.join(self.session_settings.work_dir, "%d_results.csv" % model_id)
                    with open(results_file) as in_file:
                        content = in_file.read().strip().split(",")
                        accuracy, loss = float(content[0]), float(content[1])

                    with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                        out_file.write("%s,DL,%f,%d,%d,%f,%f\n" %
                                       (self.settings.dataset, get_event_loop().time(), 0, round_nr, accuracy, loss))

                    os.unlink(results_file)

    def test_models(self, round_nr: int) -> None:
        """
        Test the accuracy of all models in the model manager locally.
        """
        for ind, model in enumerate(self.model_manager.incoming_trained_models.values()):
            print("Testing model %d on device %s..." % (ind + 1, self.settings.accuracy_device_name))
            accuracy, loss = self.evaluator.evaluate_accuracy(
                model, device_name=self.settings.accuracy_device_name)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,DL,%f,%d,%d,%f,%f\n" %
                               (self.settings.dataset, get_event_loop().time(), 0, round_nr, accuracy, loss))
