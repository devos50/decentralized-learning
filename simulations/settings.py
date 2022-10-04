from typing import List


class SimulationSettings:

    def __init__(self):
        self.peers: int = 10
        self.sample_size: int = 10
        self.batch_size: int = 500
        self.learning_rate: float = 0.002
        self.momentum: float = 0.9
        self.num_rounds: int = 10
        self.num_aggregators: int = 1
        self.duration: int = 3600  # Simulation duration in sections
        self.profile: bool = False
        self.local_classes: int = 10
        self.total_samples_per_class = 5000
        self.samples_per_class: List[int] = [self.total_samples_per_class] * 10
        self.nodes_per_class: List[int] = [self.peers] * 10
        self.dataset = "cifar10"
        self.model = "gnlenet"
        self.accuracy_logging_interval = 1
