import os
from asyncio import ensure_future

from simulations.gl.gl_simulation import GLSimulation
from simulations.settings import SimulationSettings

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 7200 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "celeba"
    settings.peers = 500
    settings.learning_rate = 0.001
    settings.sample_size = 10 if "SAMPLE_SIZE" not in os.environ else int(os.environ["SAMPLE_SIZE"])
    settings.num_aggregators = 1 if "NUM_AGGREGATORS" not in os.environ else int(os.environ["NUM_AGGREGATORS"])
    settings.fix_aggregator = bool(os.environ["FIX_AGGREGATOR"]) if "FIX_AGGREGATOR" in os.environ else False
    settings.batch_size = 20
    settings.dl_test_mode = "local" if "DL_TEST_MODE" not in os.environ else os.environ["DL_TEST_MODE"]
    settings.das_test_subprocess_jobs = 1 if "DAS_TEST_SUBPROCESS_JOBS" not in os.environ else int(os.environ["DAS_TEST_SUBPROCESS_JOBS"])
    settings.accuracy_logging_interval = 5 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.accuracy_logging_interval_is_in_sec = False if "ACC_LOG_INTERVAL_IN_SEC" not in os.environ else bool(os.environ["ACC_LOG_INTERVAL_IN_SEC"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.log_level = "INFO" if "LOG_LEVEL" not in os.environ else os.environ["LOG_LEVEL"]
    settings.latencies_file = "data/latencies.txt"
    simulation = GLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
