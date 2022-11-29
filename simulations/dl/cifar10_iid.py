import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 5
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.batch_size = 200
    settings.accuracy_logging_interval = 5 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.bypass_model_transfers = False if "BYPASS_MODEL_TRANSFERS" not in os.environ else bool(os.environ["BYPASS_MODEL_TRANSFERS"])
    settings.topology = "exp-one-peer"
    settings.latencies_file = "data/latencies.txt"
    simulation = DLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
