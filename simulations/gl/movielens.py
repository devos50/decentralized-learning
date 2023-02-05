import os
from asyncio import ensure_future

from simulations.gl.gl_simulation import GLSimulation
from simulations.settings import SimulationSettings, DLAccuracyMethod

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 * 15 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "movielens"
    settings.peers = 610
    settings.learning_rate = 0.2 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"])
    settings.batch_size = 20
    settings.dl_test_mode = "local" if "DL_TEST_MODE" not in os.environ else os.environ["DL_TEST_MODE"]
    settings.das_test_subprocess_jobs = 1 if "DAS_TEST_SUBPROCESS_JOBS" not in os.environ else int(os.environ["DAS_TEST_SUBPROCESS_JOBS"])
    settings.accuracy_logging_interval = 60 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.accuracy_logging_interval_is_in_sec = False if "ACC_LOG_INTERVAL_IN_SEC" not in os.environ else bool(os.environ["ACC_LOG_INTERVAL_IN_SEC"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.gl_round_timeout = 60 if "GL_ROUND_TIMEOUT" not in os.environ else float(os.environ["GL_ROUND_TIMEOUT"])
    settings.log_level = "INFO" if "LOG_LEVEL" not in os.environ else os.environ["LOG_LEVEL"]
    settings.latencies_file = "data/latencies.txt"
    settings.dl_accuracy_method = DLAccuracyMethod.AGGREGATE_THEN_TEST
    simulation = GLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
