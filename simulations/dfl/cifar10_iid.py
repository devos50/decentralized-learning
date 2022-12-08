import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dfl.dfl_simulation import DFLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 100 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.sample_size = 10 if "SAMPLE_SIZE" not in os.environ else int(os.environ["SAMPLE_SIZE"])
    settings.num_aggregators = 1 if "NUM_AGGREGATORS" not in os.environ else int(os.environ["NUM_AGGREGATORS"])
    settings.fix_aggregator = bool(os.environ["FIX_AGGREGATOR"]) if "FIX_AGGREGATOR" in os.environ else False
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5
    settings.latencies_file = "data/latencies.txt"
    simulation = DFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
