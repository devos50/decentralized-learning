from asyncio import ensure_future

from ipv8.configuration import ConfigBuilder

from simulations.settings import SimulationSettings
from simulations.simulation import ADFLSimulation


class BasicADFLSimulation(ADFLSimulation):

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("DFLCommunity", "my peer", [], [], {}, [])
        return builder


if __name__ == "__main__":
    settings = SimulationSettings()
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 100
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.sample_size = 10
    settings.batch_size = 20
    settings.latencies_file = "data/latencies.txt"
    simulation = BasicADFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
