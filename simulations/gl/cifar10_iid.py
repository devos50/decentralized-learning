import os
from asyncio import ensure_future

from simulations.settings import SimulationSettings
from simulations.gl.gl_simulation import GLSimulation

if __name__ == "__main__":
    settings = SimulationSettings()
    settings.duration = 3600 if "DURATION" not in os.environ else int(os.environ["DURATION"])
    settings.dataset = "cifar10"
    settings.data_distribution = "iid"
    settings.peers = 100 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])
    settings.momentum = 0.9
    settings.learning_rate = 0.002
    settings.batch_size = 20
    settings.accuracy_logging_interval = 5 if "ACC_LOG_INTERVAL" not in os.environ else int(os.environ["ACC_LOG_INTERVAL"])
    settings.checkpoint_interval = None if "CHECKPOINT_INTERVAL" not in os.environ else int(os.environ["CHECKPOINT_INTERVAL"])
    settings.train_device_name = "cpu" if "TRAIN_DEVICE_NAME" not in os.environ else os.environ["TRAIN_DEVICE_NAME"]
    settings.accuracy_device_name = "cpu" if "ACC_DEVICE_NAME" not in os.environ else os.environ["ACC_DEVICE_NAME"]
    settings.gl_round_timeout = 60 if "GL_ROUND_TIMEOUT" not in os.environ else float(os.environ["GL_ROUND_TIMEOUT"])
    settings.latencies_file = "data/latencies.txt"
    simulation = GLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()