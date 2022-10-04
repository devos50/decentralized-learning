import asyncio
import logging
import os
import shutil
import time
from asyncio import ensure_future, get_event_loop
from binascii import hexlify

import yappi

from accdfl.core.community import DFLCommunity
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import LearningSettings, DFLSettings, SessionSettings

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.settings import SimulationSettings


class ADFLSimulation:
    """
    The main logic to run simulations with ADFL.
    """

    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s" % (self.settings.peers, self.settings.dataset))
        self.peers_rounds_completed = [0] * settings.peers
        self.evaluator = None

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_overlay("DFLCommunity", "my peer", [], [], {}, [])
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

    async def on_round_complete(self, ind, round_nr):
        self.peers_rounds_completed[ind] = round_nr
        if self.settings.num_rounds and all([n >= self.settings.num_rounds for n in self.peers_rounds_completed]):
            exit(0)

    async def on_aggregate_complete(self, ind: int, round_nr: int, model):
        if round_nr % self.settings.accuracy_logging_interval == 0:
            print("Will compute accuracy!")
            accuracy, loss = self.evaluator.evaluate_accuracy(model)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%f,%d,%d,%f,%f\n" % (get_event_loop().time(), ind, round_nr, accuracy, loss))

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

    async def start_simulation(self) -> None:
        print("Starting simulation with %d peers..." % self.settings.peers)

        # Inject our nodes array in the Simulated DFL community
        for node in self.nodes:
            node.overlays[0].nodes = self.nodes

        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("time,peer,round,accuracy,loss\n")

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.settings.learning_rate,
            momentum=self.settings.momentum,
            batch_size=self.settings.batch_size
        )

        dfl_settings = DFLSettings(
            sample_size=self.settings.sample_size,
            num_aggregators=self.settings.num_aggregators,
            success_fraction=1.0,
            aggregation_timeout=2.0,
            ping_timeout=5,
            inactivity_threshold=1000
        )

        session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.settings.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=len(self.nodes),
            dfl=dfl_settings,
            data_distribution=self.settings.data_distribution,
            is_simulation=True,
        )

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, i=ind: ensure_future(self.on_round_complete(i, round_nr))
            node.overlays[0].aggregate_complete_callback = lambda round_nr, model, i=ind: self.on_aggregate_complete(i, round_nr, model)
            node.overlays[0].setup(session_settings)
            node.overlays[0].start()

        if self.settings.dataset in ["cifar10", "cifar10_niid", "mnist"]:
            data_dir = os.path.join(os.environ["HOME"], "dfl-data")
        else:
            # The LEAF dataset
            data_dir = os.path.join(os.environ["HOME"], "leaf", self.settings.dataset)

        self.evaluator = ModelEvaluator(data_dir, session_settings)

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
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.start_simulation()
        self.on_simulation_finished()
