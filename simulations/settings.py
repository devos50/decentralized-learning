class SimulationSettings:

    def __init__(self):
        self.peers: int = 10
        self.batch_size: int = 20
        self.learning_rate: float = 0.002
        self.momentum: float = 0.9
        self.duration: int = 50000  # Simulation duration in sections
        self.profile: bool = False
        self.dataset = "cifar10"
        self.accuracy_logging_interval = 1
