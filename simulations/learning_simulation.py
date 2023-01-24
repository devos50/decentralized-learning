import asyncio
import logging
import os
import shutil
import time
from statistics import median, mean
from typing import Optional

import torch

import yappi

from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings
from accdfl.dfl.community import DFLCommunity
from accdfl.dl.community import DLCommunity
from accdfl.gl.community import GLCommunity

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.dl.bypass_network_community import DLBypassNetworkCommunity
from simulations.dfl.bypass_network_community import DFLBypassNetworkCommunity
from simulations.settings import SimulationSettings


class LearningSimulation:
    """
    Base class for any simulation that involves learning.
    """

    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        self.session_settings: Optional[SessionSettings] = None
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s" % (self.settings.peers, self.settings.dataset))
        self.evaluator = None

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "curve25519", os.path.join(self.data_dir, f"ec{peer_id}.pem"))
        return builder

    async def start_ipv8_nodes(self) -> None:
        for peer_id in range(1, self.settings.peers + 1):
            if peer_id % 100 == 0:
                print("Created %d peers..." % peer_id)
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
        root.setLevel(logging.INFO)

    def ipv8_discover_peers(self) -> None:
        for node_a in self.nodes:
            for node_b in self.nodes:
                if node_a == node_b:
                    continue

                node_a.network.verified_peers.add(node_b.overlays[0].my_peer)
                node_a.network.discover_services(node_b.overlays[0].my_peer, [node_a.overlays[0].community_id, ])
        print("IPv8 peer discovery complete")

    def apply_latencies(self):
        """
        If specified in the settings, add latencies between the endpoints.
        """
        if not self.settings.latencies_file:
            return

        latencies = []
        with open(self.settings.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        print("Read latency matrix with %d sites!" % len(latencies))

        # Assign nodes to sites in a round-robin fashion and apply latencies accordingly
        for from_ind, from_node in enumerate(self.nodes):
            for to_ind, to_node in enumerate(self.nodes):
                from_site_ind = from_ind % len(latencies)
                to_site_ind = to_ind % len(latencies)
                latency_ms = int(latencies[from_site_ind][to_site_ind]) / 1000
                from_node.endpoint.latencies[to_node.endpoint.wan_address] = latency_ms

        print("Latencies applied!")

    def determine_peer_with_lowest_median_latency(self) -> int:
        """
        Based on the latencies, determine the ID of the peer with the lowest median latency to other peers.
        """
        latencies = []
        with open(self.settings.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        lowest_median_latency = 100000
        lowest_peer_id = 0
        avg_latencies = []
        for peer_id in range(min(len(self.nodes), len(latencies))):
            median_latency = median(latencies[peer_id])
            avg_latencies.append(mean(latencies[peer_id]))
            if median_latency < lowest_median_latency:
                lowest_median_latency = median_latency
                lowest_peer_id = peer_id

        print("Determined peer %d with lowest median latency: %f" % (lowest_peer_id + 1, lowest_median_latency))
        print("Average latency: %f" % mean(avg_latencies))
        return lowest_peer_id

    async def setup_simulation(self) -> None:
        print("Setting up simulation with %d peers..." % self.settings.peers)

        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("dataset,group,time,peer,round,accuracy,loss\n")

    async def start_simulation(self) -> None:
        for ind, node in enumerate(self.nodes):
            node.overlays[0].start()

        if self.settings.dataset in ["cifar10", "cifar10_niid", "mnist"]:
            data_dir = os.path.join(os.environ["HOME"], "dfl-data")
        else:
            # The LEAF dataset
            data_dir = os.path.join(os.environ["HOME"], "leaf", self.settings.dataset)

        self.evaluator = ModelEvaluator(data_dir, self.session_settings)

        if self.settings.profile:
            yappi.start(builtins=True)

        start_time = time.time()
        await asyncio.sleep(self.settings.duration)
        print("Simulation took %f seconds" % (time.time() - start_time))

        if self.settings.profile:
            yappi.stop()
            yappi_stats = yappi.get_func_stats()
            yappi_stats.sort("tsub")
            yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

        self.loop.stop()

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

    def on_simulation_finished(self) -> None:
        """
        Write away the most important data.
        """
        print("Writing away experiment statistics")

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
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.setup_simulation()
        await self.start_simulation()
        self.on_simulation_finished()
