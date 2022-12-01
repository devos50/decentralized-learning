import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dfl.dfl_simulation import DFLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 * 20 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "shakespeare_sub"
    settings.peers = 192
    settings.learning_rate = 0.8
    settings.sample_size = 10 if "SAMPLE_SIZE" not in os.environ else int(os.environ["SAMPLE_SIZE"])
    settings.num_aggregators = 1 if "NUM_AGGREGATORS" not in os.environ else int(os.environ["NUM_AGGREGATORS"])
    settings.fix_aggregator = bool(os.environ["FIX_AGGREGATOR"]) if "FIX_AGGREGATOR" in os.environ else False
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.latencies_file = "data/latencies.txt"
    simulation = DFLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
