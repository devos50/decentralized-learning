from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.simulation import ADFLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 100
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.sample_size = 10
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5
    settings.latencies_file = "data/latencies.txt"
    simulation = ADFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
