from typing import List

from accdfl.core.community import TransmissionMethod


class SimulationSettings:

    def __init__(self):
        self.peers: int = 2  # Number of IPv8 peers
        self.sample_size: int = 2
        self.batch_size: int = 20
        self.num_rounds: int = 6
        self.duration: int = 5  # Simulation duration in sections
        self.profile: bool = False
        self.local_classes: int = 10
        self.total_samples_per_class = 1000
        self.samples_per_class: List[int] = [self.total_samples_per_class] * 10
        self.nodes_per_class: List[int] = [self.peers] * 10
        self.dataset = "mnist"
        self.model = "linear"
        self.transmission_method = TransmissionMethod.EVA
