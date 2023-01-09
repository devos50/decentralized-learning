from dataclasses import dataclass
from typing import List, Optional, Type

from dataclasses_json import dataclass_json

from accdfl.core import TransmissionMethod
from accdfl.core.gradient_aggregation import GradientAggregationMethod
from accdfl.core.gradient_aggregation.fedavg import FedAvg


@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    batch_size: int
    kd_temperature: Optional[float] = None
    kd_alpha: Optional[float] = None


@dataclass
class DFLSettings:
    """
    Setting related to sample-based decentralized federated learning.
    """
    sample_size: int
    num_aggregators: int
    success_fraction: float = 1
    aggregation_timeout: float = 5
    ping_timeout: float = 5
    inactivity_threshold: int = 50
    fixed_aggregator: Optional[bytes] = None


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
    dfl: Optional[DFLSettings] = None
    dl: Optional[DLSettings] = None
    gl: Optional[GLSettings] = None
    data_distribution: str = "iid"
    gradient_aggregation: GradientAggregationMethod = GradientAggregationMethod.FEDAVG
    model_seed: int = 0
    model_send_delay: float = 1.0
    train_in_subprocess: bool = False
    transmission_method: TransmissionMethod = TransmissionMethod.EVA
    eva_block_size: int = 60000  # This value is extremely high and tuned for the DAS6
    eva_max_simultaneous_transfers: int = 30  # Corresponds to a peak usage of ~3.4 MB/s for an aggregator
    is_simulation: bool = False
    train_device_name: str = "cpu"
