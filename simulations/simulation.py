import asyncio
import logging
import os
import shutil
import time

import yappi

from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings

from ipv8.configuration import ConfigBuilder
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.dfl.community import SimulatedDFLCommunity
from simulations.gl.community import SimulatedGLCommunity

from simulations.settings import SimulationSettings


class DLSimulation:
    """
    The main logic to run simulations.
    """

    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d" % self.settings.peers)
        self.peers_rounds_completed = [0] * settings.peers
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
                            extra_communities={'SimulatedDFLCommunity': SimulatedDFLCommunity,
                                               'SimulatedGLCommunity': SimulatedGLCommunity})
            await instance.start()
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

                node_a.overlays[0].walk_to(node_b.overlays[0].my_peer.address)
        print("IPv8 peer discovery complete")

    async def on_aggregate_complete(self, ind: int, round_nr: int, model):
        if round_nr % self.settings.accuracy_logging_interval == 0:
            print("Will compute accuracy!")
            accuracy, loss = self.evaluator.evaluate_accuracy(model)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%d,%f,%d,%f,%f\n" % (ind, asyncio.get_event_loop().time(), round_nr, accuracy, loss))

    def get_session_settings(self) -> SessionSettings:
        raise NotImplementedError

    async def start_simulation(self) -> None:
        print("Starting simulation with %d peers..." % self.settings.peers)

        # Inject our nodes array in the Simulated DFL community
        for node in self.nodes:
            node.overlays[0].nodes = self.nodes

        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("peer,time,step,accuracy,loss\n")

        settings = self.get_session_settings()
        for ind, node in enumerate(self.nodes):
            node.overlays[0].aggregate_complete_callback = lambda round_nr, model, i=ind: self.on_aggregate_complete(i, round_nr, model)
            node.overlays[0].setup(settings)
            node.overlays[0].start()

        self.evaluator = ModelEvaluator(os.path.join(os.environ["HOME"], "dfl-data"), settings)

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
