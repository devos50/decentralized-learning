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
    client_learning_rate: float
    client_optimizer: str
    server_learning_rate: float
    server_optimizer: str
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
    topology: str = "k-regular"
    el: bool = False
    k: int = 2


@dataclass
class TeleportationSettings:
    """
    Setting related to teleportation.
    """
    topology: str = "k-regular"
    k: int = 2
    sample_size: int = 1


@dataclass
class DiLoCoSettings:
    """
    Setting related to DiLoCo.
    """
    pass


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
    teleportation: Optional[TeleportationSettings] = None
    diloco: Optional[DiLoCoSettings] = None
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
