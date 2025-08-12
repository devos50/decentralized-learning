import os
from dataclasses import dataclass
from typing import List, Optional

from dataclasses_json import dataclass_json

from accdfl.core import TransmissionMethod


@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    local_steps: int


@dataclass
class DFLSettings:
    """
    Setting related to sample-based decentralized federated learning.
    """
    sample_size: int
    num_aggregators: int
    success_fraction: float = 1
    liveness_success_fraction: float = 0.4
    ping_timeout: float = 5
    inactivity_threshold: int = 50
    fixed_aggregator: Optional[bytes] = None
    aggregation_timeout: float = 300


@dataclass
class DLSettings:
    """
    Setting related to decentralized learning.
    """
    topology: str = "exp-one-peer"


@dataclass
class GLSettings:
    """
    Setting related to gossip learning.
    """
    round_timeout: float = 60


@dataclass_json
@dataclass
class SessionSettings:
    """
    All settings related to a training session.
    """
    work_dir: str
    dataset: str
    learning: LearningSettings
    participants: List[str]
    all_participants: List[str]
    target_participants: int
    dataset_base_path: str = None
    dfl: Optional[DFLSettings] = None
    dl: Optional[DLSettings] = None
    gl: Optional[GLSettings] = None
    model: Optional[str] = None
    alpha: float = 1
    partitioner: str = "uniform"  # uniform or dirichlet
    model_seed: int = 0
    model_send_delay: float = 1.0
    transmission_method: TransmissionMethod = TransmissionMethod.EVA
    eva_block_size: int = 60000  # This value is extremely high and tuned for the DAS6
    eva_max_simultaneous_transfers: int = 30  # Corresponds to a peak usage of ~3.4 MB/s for an aggregator
    bypass_training: bool = False  # Whether to bypass model training, can be useful to observe network dynamics
    device: str = "cpu"
    aggregate: str = "fedavg"  # fedavg or fedadam
