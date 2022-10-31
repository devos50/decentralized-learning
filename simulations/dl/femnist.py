import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 * 4 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "femnist"
    settings.peers = 355
    settings.learning_rate = 0.004
    settings.batch_size = 20
    settings.accuracy_logging_interval = 3
    settings.topology = "exp-one-peer"
    settings.latencies_file = "data/latencies.txt"
    simulation = DLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
