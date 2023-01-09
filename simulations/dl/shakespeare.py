import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 * 50 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "shakespeare"
    settings.peers = 192 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])
    settings.learning_rate = 0.2 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"])
    settings.num_aggregators = 1 if "NUM_AGGREGATORS" not in os.environ else int(os.environ["NUM_AGGREGATORS"])
    settings.fix_aggregator = bool(os.environ["FIX_AGGREGATOR"]) if "FIX_AGGREGATOR" in os.environ else False
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.bypass_model_transfers = False if "BYPASS_MODEL_TRANSFERS" not in os.environ else bool(os.environ["BYPASS_MODEL_TRANSFERS"])
    settings.latencies_file = "data/latencies.txt"
    settings.topology = "exp-one-peer"
    simulation = DLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()