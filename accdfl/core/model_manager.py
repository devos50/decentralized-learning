import logging
from typing import Dict, Optional

from peft import PeftModel
import torch

from accdfl.core.gradient_aggregation import GradientAggregationMethod
from accdfl.core.gradient_aggregation.fedadam import FedAdam
from accdfl.core.gradient_aggregation.fedavg import FedAvg
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, peft_model: Optional[PeftModel], settings: SessionSettings, participant_index: int):
        self.peft_model: PeftModel = peft_model
        self.adapter: Dict = None
        self.global_adapter: Dict = None
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_trainer: ModelTrainer = ModelTrainer(self.settings, self.participant_index)
        self.aggregator: Optional[FedAdam] = None

        # Keeps track of the incoming trained adapters as aggregator
        self.incoming_trained_adapters: Dict[bytes, Dict] = {}

    def process_incoming_trained_adapter(self, peer_pk: bytes, incoming_adapter: Dict):
        if peer_pk in self.incoming_trained_adapters:
            # We already processed this adapter
            return

        self.incoming_trained_adapters[peer_pk] = incoming_adapter

    def reset_incoming_trained_adapters(self):
        self.incoming_trained_adapters = {}

    def aggregate_trained_adapters(self):
        adapters = [adapter for adapter in self.incoming_trained_adapters.values()]
        self.aggregator.aggregate(adapters, self.global_adapter, self.peft_model)

    async def train(self) -> int:
        samples_trained_on = await self.model_trainer.train(self.peft_model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples_trained_on

    def adopt_adapter(self, new_adapter: Dict):
        """
        Adopt the given adapter as the current model adapter.
        """
        model_state_dict = self.peft_model.state_dict()

        for key in new_adapter["keys"]:
            # Get the associated key in our adapter
            our_key = key.replace(new_adapter["name"], self.adapter["name"])
            if our_key in self.adapter["keys"]:
                model_state_dict[our_key].copy_(model_state_dict[key])
            else:
                self.logger.warning(f"Key {key} not found in our adapter, keeping the existing value.")
