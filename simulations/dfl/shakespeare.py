from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.simulation import ADFLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.dataset = "shakespeare_sub"
    settings.peers = 192
    settings.learning_rate = 0.8
    settings.sample_size = 10
    settings.batch_size = 20
    settings.latencies_file = "data/latencies.txt"
    simulation = ADFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
