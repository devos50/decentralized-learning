import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.simulation import ADFLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.dataset = "movielens"
    settings.duration = 3600 * 4
    settings.peers = 610
    settings.learning_rate = 0.25
    settings.sample_size = 10 if "SAMPLE_SIZE" not in os.environ else int(os.environ["SAMPLE_SIZE"])
    settings.num_aggregators = 1 if "NUM_AGGREGATORS" not in os.environ else int(os.environ["NUM_AGGREGATORS"])
    settings.fix_aggregator = bool(os.environ["FIX_AGGREGATOR"]) if "FIX_AGGREGATOR" in os.environ else False
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5
    settings.latencies_file = "data/latencies.txt"
    simulation = ADFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()