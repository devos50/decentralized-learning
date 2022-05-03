import copy
import logging
import os
from asyncio import get_event_loop
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, List

import torch
import torch.nn as nn


from accdfl.core.evaluator import setup_evaluator, evaluate_accuracy
from accdfl.core.model_trainer import setup_trainer, train_model, ModelTrainer


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model, parameters, participant_index: int):
        self.model = model
        self.epoch = 1
        self.parameters = parameters
        self.participant_index = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = os.path.join(os.environ["HOME"], "dfl-data")
        self.acc_check_executor = ProcessPoolExecutor(initializer=setup_evaluator,
                                                      initargs=(self.data_dir, parameters,),
                                                      max_workers=4)
        self.model_train_executor = ProcessPoolExecutor(initializer=setup_trainer,
                                                        initargs=(self.data_dir, parameters, participant_index,),
                                                        max_workers=1)
        self.model_trainer = None  # Only used when not training in a subprocess (e.g., in the unit tests)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[int, Dict[bytes, nn.Module]] = {}

    def process_incoming_trained_model(self, peer_pk: bytes, round: int, incoming_model: nn.Module):
        if round in self.incoming_trained_models and peer_pk in self.incoming_trained_models[round]:
            # We already processed this model
            return

        if round not in self.incoming_trained_models:
            self.incoming_trained_models[round] = {}
        self.incoming_trained_models[round][peer_pk] = incoming_model

    def has_enough_trained_models_of_round(self, round: int) -> bool:
        if round not in self.incoming_trained_models:
            return False
        return len(self.incoming_trained_models[round]) >= (self.parameters["sample_size"] * self.parameters["success_fraction"])

    def average_trained_models_of_round(self, round: int) -> Optional[nn.Module]:
        if round not in self.incoming_trained_models:
            return None
        models = [model for model in self.incoming_trained_models[round].values()]
        return self.average_models(models)

    def remove_trained_models_of_round(self, round: int) -> None:
        self.incoming_trained_models.pop(round)

    async def train(self, in_subprocess: bool = True):
        if in_subprocess:
            trained_model = await get_event_loop().run_in_executor(self.model_train_executor, train_model, self.model)
        else:
            if not self.model_trainer:
                # Lazy initialize the model trainer
                self.model_trainer = ModelTrainer(self.data_dir, self.parameters, self.participant_index)
            trained_model = self.model_trainer.train(self.model)
        self.model = trained_model

    @staticmethod
    def average_models(models: List[nn.Module]) -> nn.Module:
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model

    async def compute_accuracy(self, model: nn.Module):
        """
        Compute the accuracy/loss of the current model.
        Optionally, one can provide a custom iterator to compute the accuracy on a custom dataset.
        """
        self.logger.info("Computing accuracy of model")
        accuracy, loss = await get_event_loop().run_in_executor(self.acc_check_executor, evaluate_accuracy, model)
        return accuracy, loss
