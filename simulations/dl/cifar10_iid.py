from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 100
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.batch_size = 20
    settings.latencies_file = "data/latencies.txt"
    simulation = DLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
