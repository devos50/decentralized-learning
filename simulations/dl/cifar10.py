import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "cifar10"
    settings.alpha = 1 if "ALPHA" not in os.environ else float(os.environ["ALPHA"])
    settings.model = None if "MODEL" not in os.environ else os.environ["MODEL"]
    settings.partitioner = "dirichlet" if "PARTITIONER" not in os.environ else os.environ["PARTITIONER"]
    settings.peers = 100 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])
    settings.momentum = 0.9 if "MOMENTUM" not in os.environ else float(os.environ["MOMENTUM"])
    settings.learning_rate = 0.002 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"])
    settings.batch_size = 20
    settings.dl_test_mode = "local" if "DL_TEST_MODE" not in os.environ else os.environ["DL_TEST_MODE"]
    settings.das_test_subprocess_jobs = 1 if "DAS_TEST_SUBPROCESS_JOBS" not in os.environ else int(os.environ["DAS_TEST_SUBPROCESS_JOBS"])
    settings.accuracy_logging_interval = 5 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.bypass_model_transfers = False if "BYPASS_MODEL_TRANSFERS" not in os.environ else bool(os.environ["BYPASS_MODEL_TRANSFERS"])
    settings.topology = "exp-one-peer"
    settings.latencies_file = "data/latencies.txt"
    simulation = DLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
