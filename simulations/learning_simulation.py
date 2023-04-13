import asyncio
import logging
import os
import pickle
import shutil
import stat
import subprocess
import time
from argparse import Namespace
from statistics import median, mean
from typing import Dict, List, Optional, Tuple

import torch

import yappi

from accdfl.core.model_manager import ModelManager
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings, dump_settings
from accdfl.dfl.community import DFLCommunity
from accdfl.dl.community import DLCommunity
from accdfl.gl.community import GLCommunity

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.dl.bypass_network_community import DLBypassNetworkCommunity
from simulations.dfl.bypass_network_community import DFLBypassNetworkCommunity


class LearningSimulation:
    """
    Base class for any simulation that involves learning.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.session_settings: Optional[SessionSettings] = None
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s" % (self.args.peers, self.args.dataset))
        self.evaluator = None
        self.logger = None
        self.model_manager: Optional[ModelManager] = None

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "curve25519", os.path.join(self.data_dir, f"ec{peer_id}.pem"))
        return builder

    async def start_ipv8_nodes(self) -> None:
        for peer_id in range(1, self.args.peers + 1):
            if peer_id % 100 == 0:
                self.logger.info("Created %d peers..." % peer_id)
            endpoint = SimulationEndpoint()
            instance = IPv8(self.get_ipv8_builder(peer_id).finalize(), endpoint_override=endpoint,
                            extra_communities={
                                'DLCommunity': DLCommunity,
                                'GLCommunity': GLCommunity,
                                'DLBypassNetworkCommunity': DLBypassNetworkCommunity,
                                'DFLCommunity': DFLCommunity,
                                'DFLBypassNetworkCommunity': DFLBypassNetworkCommunity,
                            })
            await instance.start()

            # Set the WAN address of the peer to the address of the endpoint
            for overlay in instance.overlays:
                overlay.max_peers = -1
                overlay.my_peer.address = instance.overlays[0].endpoint.wan_address
                overlay.my_estimated_wan = instance.overlays[0].endpoint.wan_address

            self.nodes.append(instance)

    def setup_directories(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_logger(self) -> None:
        root = logging.getLogger()
        root.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        root.setLevel(getattr(logging, self.args.log_level))

        self.logger = logging.getLogger(self.__class__.__name__)

    def ipv8_discover_peers(self) -> None:
        for node_a in self.nodes:
            for node_b in self.nodes:
                if node_a == node_b:
                    continue

                node_a.network.verified_peers.add(node_b.overlays[0].my_peer)
                node_a.network.discover_services(node_b.overlays[0].my_peer, [node_a.overlays[0].community_id, ])
        self.logger.info("IPv8 peer discovery complete")

    def apply_traces(self):
        """
        Set the relevant traces.
        """
        if self.args.availability_traces:
            self.logger.info("Applying availability trace file %s", self.args.availability_traces)
            with open(self.args.availability_traces, "rb") as traces_file:
                data = pickle.load(traces_file)

            # TODO we simply pick the first n devices from the list for now
            for ind, node in enumerate(self.nodes):
                node.overlays[0].set_traces(data[ind + 1])

        self.logger.info("Traces applied!")

    def apply_latencies(self):
        """
        If specified in the settings, add latencies between the endpoints.
        """
        if not self.args.latencies_file:
            return

        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        self.logger.info("Read latency matrix with %d sites!" % len(latencies))

        # Assign nodes to sites in a round-robin fashion and apply latencies accordingly
        for from_ind, from_node in enumerate(self.nodes):
            for to_ind, to_node in enumerate(self.nodes):
                from_site_ind = from_ind % len(latencies)
                to_site_ind = to_ind % len(latencies)
                latency_ms = int(latencies[from_site_ind][to_site_ind]) / 1000
                from_node.endpoint.latencies[to_node.endpoint.wan_address] = latency_ms

        self.logger.info("Latencies applied!")

    def determine_peer_with_lowest_median_latency(self, eligible_peers: List[int]) -> int:
        """
        Based on the latencies, determine the ID of the peer with the lowest median latency to other peers.
        """
        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        lowest_median_latency = 100000
        lowest_peer_id = 0
        avg_latencies = []
        for peer_id in range(min(len(self.nodes), len(latencies))):
            if peer_id not in eligible_peers:
                continue
            median_latency = median(latencies[peer_id])
            avg_latencies.append(mean(latencies[peer_id]))
            if median_latency < lowest_median_latency:
                lowest_median_latency = median_latency
                lowest_peer_id = peer_id

        self.logger.info("Determined peer %d with lowest median latency: %f", lowest_peer_id + 1, lowest_median_latency)
        self.logger.debug("Average latency: %f" % mean(avg_latencies))
        return lowest_peer_id

    async def setup_simulation(self) -> None:
        self.logger.info("Setting up simulation with %d peers..." % self.args.peers)
        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("dataset,group,time,peer,round,accuracy,loss\n")

    async def start_simulation(self) -> None:
        nodes_started: int = 0
        for ind, node in enumerate(self.nodes):
            if node.overlays[0].traces and node.overlays[0].traces["active"][0] == 0:
                node.overlays[0].start()
                nodes_started += 1
        self.logger.info("Started %d nodes...", nodes_started)

        if self.args.dataset in ["cifar10", "mnist"]:
            data_dir = os.path.join(os.environ["HOME"], "dfl-data")
        else:
            # The LEAF dataset
            data_dir = os.path.join(os.environ["HOME"], "leaf", self.args.dataset)

        self.evaluator = ModelEvaluator(data_dir, self.session_settings)

        if self.args.profile:
            yappi.start(builtins=True)

        if self.args.profile:
            yappi.stop()
            yappi_stats = yappi.get_func_stats()
            yappi_stats.sort("tsub")
            yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

        start_time = time.time()
        if self.args.duration > 0:
            await asyncio.sleep(self.args.duration)
            self.logger.info("Simulation took %f seconds" % (time.time() - start_time))
            self.loop.stop()
        else:
            self.logger.info("Running simulation for undefined time")

    def on_ipv8_ready(self) -> None:
        """
        This method is called when IPv8 is started and peer discovery is finished.
        """
        pass

    def checkpoint_models(self, round_nr: int):
        """
        Dump all models during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        shutil.rmtree(models_dir, ignore_errors=True)
        os.makedirs(models_dir, exist_ok=True)

        avg_model = self.model_manager.aggregate_trained_models()
        for peer_ind, node in enumerate(self.nodes):
            torch.save(node.overlays[0].model_manager.model.state_dict(),
                       os.path.join(models_dir, "%d.model" % peer_ind))
        torch.save(avg_model.state_dict(), os.path.join(models_dir, "avg.model"))

    def checkpoint_model(self, peer_ind: int, round_nr: int):
        """
        Checkpoint a particular model of a peer during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        os.makedirs(models_dir, exist_ok=True)

        model = self.nodes[peer_ind].overlays[0].model_manager.model
        torch.save(model.state_dict(), os.path.join(models_dir, "%d.model" % peer_ind))

    def test_models_with_das_jobs(self) -> Dict[int, Tuple[float, float]]:
        """
        Test the accuracy of all models in the model manager by spawning different DAS jobs.
        """
        results: Dict[int, Tuple[float, float]] = {}

        dump_settings(self.session_settings)

        # Divide the clients over the DAS nodes
        client_queue = list(range(len(self.model_manager.incoming_trained_models.values())))
        while client_queue:
            self.logger.info("Scheduling new batch on DAS nodes - %d clients left", len(client_queue))

            processes = []
            all_model_ids = set()
            for job_ind in range(self.args.das_test_subprocess_jobs):
                if not client_queue:
                    continue

                clients_on_this_node = []
                while client_queue and len(clients_on_this_node) < self.args.das_test_num_models_per_subprocess:
                    client = client_queue.pop(0)
                    clients_on_this_node.append(client)

                # Prepare the input files for the subjobs
                model_ids = []
                for client_id in clients_on_this_node:
                    model_id = client_id
                    model_ids.append(model_id)
                    all_model_ids.add(model_id)
                    model_file_name = "%d.model" % model_id
                    model_path = os.path.join(self.session_settings.work_dir, model_file_name)
                    node_id = b"%d" % client_id
                    torch.save(self.model_manager.incoming_trained_models[node_id].state_dict(), model_path)

                import accdfl.util as autil
                script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "evaluate_model.py")

                # Prepare the files and spawn the processes!
                out_file_path = os.path.join(os.getcwd(), "out_%d.log" % job_ind)
                model_ids_str = ",".join(["%d" % model_id for model_id in model_ids])

                train_cmd = "python3 %s %s %s %s" % (
                script_dir, self.session_settings.work_dir, model_ids_str, self.model_manager.data_dir)
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

                cmd = "ssh fs3.das6.tudelft.nl \"prun -t 5:00 -np 1 -o %s %s\"" % (
                out_file_path, os.path.join(os.getcwd(), bash_file_name))
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
                        results[model_id] = (accuracy, loss)

                    os.unlink(results_file)

        return results

    def test_models(self) -> Dict[int, Tuple[float, float]]:
        """
        Test the accuracy of all models in the model manager locally.
        """
        results: Dict[int, Tuple[float, float]] = {}
        for ind, model in enumerate(self.model_manager.incoming_trained_models.values()):
            self.logger.info("Testing model %d on device %s..." % (ind + 1, self.args.accuracy_device_name))
            accuracy, loss = self.evaluator.evaluate_accuracy(
                model, device_name=self.args.accuracy_device_name)
            results[ind] = (accuracy, loss)
        return results

    def on_simulation_finished(self) -> None:
        """
        Write away the most important data.
        """
        self.logger.info("Writing away experiment statistics")

        # Write away the model transfer times
        with open(os.path.join(self.data_dir, "transfer_times.csv"), "w") as transfer_times_file:
            transfer_times_file.write("time\n")
            for node in self.nodes:
                for transfer_time in node.overlays[0].transfer_times:
                    transfer_times_file.write("%f\n" % transfer_time)

        # Write away the model training times
        with open(os.path.join(self.data_dir, "training_times.csv"), "w") as training_times_file:
            training_times_file.write("peer,duration\n")
            for ind, node in enumerate(self.nodes):
                for training_time in node.overlays[0].model_manager.training_times:
                    training_times_file.write("%d,%f\n" % (ind + 1, training_time))

        # Write away the individual, generic bandwidth statistics
        tot_up, tot_down = 0, 0
        with open(os.path.join(self.data_dir, "bandwidth.csv"), "w") as bw_file:
            bw_file.write("peer,outgoing_bytes,incoming_bytes\n")
            for ind, node in enumerate(self.nodes):
                tot_up += node.overlays[0].endpoint.bytes_up
                tot_down += node.overlays[0].endpoint.bytes_down
                bw_file.write("%d,%d,%d\n" % (ind + 1,
                                              node.overlays[0].endpoint.bytes_up,
                                              node.overlays[0].endpoint.bytes_down))

        # Write away the total, generic bandwidth statistics
        with open(os.path.join(self.data_dir, "total_bandwidth.txt"), "w") as bw_file:
            bw_file.write("%d,%d" % (tot_up, tot_down))

    async def run(self) -> None:
        self.setup_directories()
        await self.start_ipv8_nodes()
        self.setup_logger()
        self.ipv8_discover_peers()
        self.apply_traces()
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.setup_simulation()
        await self.start_simulation()
        self.on_simulation_finished()
