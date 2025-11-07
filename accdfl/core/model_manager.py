import logging
from typing import Dict, Optional

import torch
from transformers import PreTrainedModel

from accdfl.core.gradient_aggregation import GradientAggregation
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import SessionSettings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model: PreTrainedModel, settings: SessionSettings, participant_index: int):
        self.model: PreTrainedModel = model
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_trainer: ModelTrainer = ModelTrainer(self.settings, self.participant_index)
        self.server_optimizer: Optional[GradientAggregation] = None

    def process_incoming_trained_adapter(self, peer_pk: bytes, incoming_adapter: Dict):
        if peer_pk in self.incoming_trained_adapters:
            # We already processed this adapter
            return

        self.incoming_trained_adapters[peer_pk] = incoming_adapter

    def reset_incoming_trained_adapters(self):
        self.incoming_trained_adapters = {}

    def aggregate_trained_adapters(self):
        adapters = [adapter for adapter in self.incoming_trained_adapters.values()]
        self.aggregator.aggregate(adapters)

    async def train(self) -> int:
        samples_trained_on = await self.model_trainer.train(self.model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples_trained_on
    
    def apply_outer_optimizer(self, aggregated_gradients: list):
        self.model_trainer.apply_outer_optimizer(self.model, aggregated_gradients, self.server_optimizer)
