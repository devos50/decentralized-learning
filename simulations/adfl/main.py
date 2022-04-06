from asyncio import ensure_future

from ipv8.configuration import ConfigBuilder

from simulations.settings import SimulationSettings
from simulations.simulation import ADFLSimulation


class BasicADFLSimulation(ADFLSimulation):

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("DFLCommunity", "my peer", [], [], {"working_directory": ":memory:"}, [])
        return builder


if __name__ == "__main__":
    settings = SimulationSettings()
    settings.peers = 2
    settings.duration = 20
    simulation = BasicADFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
