import asyncio
import logging
import os
import random
import shutil
import time
from binascii import hexlify

import yappi

from accdfl.core.community import DFLCommunity, TransmissionMethod

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.settings import SimulationSettings

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms


class ADFLSimulation:
    """
    The main logic to run simulations with ADFL.
    """

    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d" % self.settings.peers)
        self.peers_rounds_completed = [0] * settings.peers

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
                            extra_communities={'DFLCommunity': DFLCommunity})
            await instance.start()
            self.nodes.append(instance)

    def setup_directories(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_logger(self) -> None:
        root = logging.getLogger()
        root.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        root.setLevel(logging.WARN)

    def ipv8_discover_peers(self) -> None:
        for node_a in self.nodes:
            connect_nodes = random.sample(self.nodes, min(100, len(self.nodes)))
            for node_b in connect_nodes:
                if node_a == node_b:
                    continue

                node_a.overlays[0].walk_to(node_b.endpoint.wan_address)
        print("IPv8 peer discovery complete")

    async def on_round_complete(self, ind, round_nr):
        self.peers_rounds_completed[ind] = round_nr

        if ind == 0 and round_nr % self.settings.accuracy_logging_interval == 0:
            accuracy, loss = await self.nodes[0].overlays[0].compute_accuracy(include_wait_periods=False)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%d,%d,%f,%f\n" % (ind, round_nr, accuracy, loss))

        if all([n >= self.settings.num_rounds for n in self.peers_rounds_completed]):
            exit(0)

    async def start_simulation(self) -> None:
        print("Starting simulation with %d peers..." % self.settings.peers)

        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("peer,step,accuracy,loss\n")

        # Setup the training process
        experiment_data = {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "learning_rate": self.settings.learning_rate,
            "momentum": self.settings.momentum,
            "batch_size": self.settings.batch_size,
            "participants": [hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            "rounds": self.settings.num_rounds,
            "sample_size": self.settings.sample_size,

            # These parameters are not available in a deployed environment - only for experimental purposes.
            "samples_per_class": self.settings.samples_per_class,
            "local_classes": self.settings.local_classes,
            "nodes_per_class": self.settings.nodes_per_class,
            "dataset": self.settings.dataset,
            "model": self.settings.model,
        }
        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, _, i=ind: self.on_round_complete(i, round_nr)
            node.overlays[0].setup(experiment_data, None, transmission_method=self.settings.transmission_method)
            node.overlays[0].start()

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

    def on_simulation_finished(self) -> None:
        """
        This method is called when the simulations are finished.
        """
        pass

    async def run(self) -> None:
        self.setup_directories()
        await self.start_ipv8_nodes()
        self.setup_logger()
        self.ipv8_discover_peers()
        self.on_ipv8_ready()
        await self.start_simulation()
        self.on_simulation_finished()
