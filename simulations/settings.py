from typing import Optional


class SimulationSettings:

    def __init__(self):
        self.peers: int = 10
        self.sample_size: int = 1
        self.batch_size: int = 500
        self.learning_rate: float = 0.002
        self.momentum: float = 0.0
        self.num_rounds: Optional[int] = None
        self.num_aggregators: int = 1
        self.duration: int = 3600  # Simulation duration in seconds
        self.profile: bool = False
        self.dataset: str = "cifar10"
        self.data_distribution: str = "iid"
        self.accuracy_logging_interval: int = 1
        self.latencies_file: Optional[str] = None
