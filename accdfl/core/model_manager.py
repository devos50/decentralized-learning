import logging
from typing import Dict, Optional

from peft import PeftModel, get_peft_model_state_dict
import torch

from accdfl.core.gradient_aggregation import GradientAggregationMethod
from accdfl.core.gradient_aggregation.fedavg import FedAvg
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, peft_model: Optional[PeftModel], settings: SessionSettings, participant_index: int):
        self.peft_model: PeftModel = peft_model
        self.adapter: Dict = get_peft_model_state_dict(peft_model, adapter_name=f"client_{participant_index}")
        self.global_adapter: Dict = get_peft_model_state_dict(peft_model, adapter_name=f"global")
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_trainer: ModelTrainer = ModelTrainer(self.settings, self.participant_index)

        # Keeps track of the incoming trained adapters as aggregator
        self.incoming_trained_adapters: Dict[bytes, Dict] = {}

    def process_incoming_trained_adapter(self, peer_pk: bytes, incoming_adapter: Dict):
        if peer_pk in self.incoming_trained_adapters:
            # We already processed this adapter
            return

        self.incoming_trained_adapters[peer_pk] = incoming_adapter

    def reset_incoming_trained_adapters(self):
        self.incoming_trained_adapters = {}

    def get_aggregation_method(self):
        if self.settings.gradient_aggregation == GradientAggregationMethod.FEDAVG:
            return FedAvg

    def aggregate_trained_adapters(self) -> Dict:
        adapters = [adapter for adapter in self.incoming_trained_adapters.values()]
        return self.get_aggregation_method().aggregate(adapters, self.peft_model)

    async def train(self) -> int:
        samples_trained_on = await self.model_trainer.train(self.peft_model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples_trained_on

    def adopt_adapter(self, new_adapter: Dict):
        """
        Adopt the given adapter as the current model adapter.
        """
        for key in self.adapter.keys():
            if key in new_adapter:
                self.adapter[key].copy_(new_adapter[key])
            else:
                self.logger.warning(f"Key {key} not found in the incoming adapter, keeping the existing value.")
