from enum import Enum
from typing import Optional


class DLAccuracyMethod(Enum):
    """
    The method used to determine the final accuracy for DL.
    """
    AGGREGATE_THEN_TEST = 0
    TEST_INDIVIDUAL_MODELS = 1


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
        self.alpha: float = 1  # Related to the Dirichlet data distributor
        self.model: Optional[str] = None
        self.data_distribution: str = "iid"
        self.accuracy_logging_interval: int = 1
        self.dl_accuracy_method: DLAccuracyMethod = DLAccuracyMethod.TEST_INDIVIDUAL_MODELS
        self.train_device_name: str = "cpu"
        self.accuracy_device_name: str = "cpu"
        self.checkpoint_interval: Optional[int] = None
        self.latencies_file: Optional[str] = None
        self.fix_aggregator: bool = False
        self.topology: Optional[str] = None
        self.bypass_model_transfers: bool = False
        self.gl_round_timeout: float = 60
