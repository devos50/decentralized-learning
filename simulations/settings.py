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
        self.active_participants: Optional[str] = None  # In the form "5-23"
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
        self.partitioner: str = "iid"
        self.accuracy_logging_interval: int = 1
        self.accuracy_logging_interval_is_in_sec: bool = False
        self.dl_accuracy_method: DLAccuracyMethod = DLAccuracyMethod.TEST_INDIVIDUAL_MODELS
        self.dl_test_mode: str = "local"  # How we test the models, options: local and das_jobs
        self.das_test_subprocess_jobs: int = 1  # The number of subprocesses we should spawn to evaluate the models
        self.das_test_num_models_per_subprocess: int = 10  # The number of models we test per subprocess
        self.train_device_name: str = "cpu"
        self.accuracy_device_name: str = "cpu"
        self.checkpoint_interval: Optional[int] = None
        self.store_best_models: bool = False
        self.latencies_file: Optional[str] = None
        self.fix_aggregator: bool = False
        self.topology: Optional[str] = None
        self.bypass_model_transfers: bool = False
        self.gl_round_timeout: float = 60
        self.log_level: str = "INFO"
