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
    batch_size: int


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
    data_distribution: str = "iid"
    model_seed: int = 0
    model_send_delay: float = 1.0
    train_in_subprocess: bool = False
    transmission_method: TransmissionMethod = TransmissionMethod.EVA
    eva_block_size: int = 60000  # This value is extremely high and tuned for the DAS6
    is_simulation: bool = False
