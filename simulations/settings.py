from typing import List

from accdfl.core.community import TransmissionMethod


class SimulationSettings:

    def __init__(self):
        self.peers: int = 2  # Number of IPv8 peers
        self.sample_size: int = 2
        self.batch_size: int = 200
        self.learning_rate: float = 0.1
        self.momentum: float = 0
        self.num_rounds: int = 100
        self.duration: int = 50000  # Simulation duration in sections
        self.profile: bool = False
        self.local_classes: int = 10
        self.total_samples_per_class = 5000
        self.samples_per_class: List[int] = [self.total_samples_per_class] * 10
        self.nodes_per_class: List[int] = [self.peers] * 10
        self.dataset = "mnist"
        self.model = "gnlenet"
        self.transmission_method = TransmissionMethod.EVA
        self.accuracy_logging_interval = 25
