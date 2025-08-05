from asyncio import CancelledError, get_event_loop, sleep
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from accdfl.core.session_settings import SessionSettings

AUGMENTATION_FACTOR_SIM = 3.0


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, settings: SessionSettings, participant_index: int):
        """
        :param simulated_speed: compute speed of the simulated device, in ms/sample.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.simulated_speed: Optional[float] = None
        self.total_training_time: float = 0
        self.is_training: bool = False
        self.dataset: Optional[Dataset] = None
        self.tokenizer: Optional[AutoTokenizer] = None
    
    def setup_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

    async def train(self, peft_model) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many samples were trained on.
        """
        self.is_training = True
        samples_trained_on = 0

        # If we're running a simulation, we should advance the time of the DiscreteLoop with either the simulated
        # elapsed time or the elapsed real-world time for training. Otherwise,training would be considered instant
        # in our simulations. We do this before the actual training so if our sleep gets interrupted, the local
        # model will not be updated.
        start_time = get_event_loop().time()
        if self.simulated_speed:
            elapsed_time = AUGMENTATION_FACTOR_SIM * self.settings.learning.local_steps * (self.simulated_speed / 1000)
        else:
            elapsed_time = 0

        try:
            await sleep(elapsed_time)
        except CancelledError:
            self.is_training = False
            self.total_training_time += (get_event_loop().time() - start_time)
            return 0  # Training got interrupted - don't update the model
        self.total_training_time += elapsed_time

        peft_model.set_adapter("client_%d" % self.participant_index)
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=self.settings.learning.learning_rate)
        peft_model.train()
        train_dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True, collate_fn=self.data_collator)
        train_set_it = iter(train_dataloader)

        for local_step in range(self.settings.learning.local_steps):
            batch = next(train_set_it)
            optimizer.zero_grad()
            inputs = {k: v.to(self.settings.device) for k, v in batch.items() if k != 'labels'}
            samples_trained_on += len(batch['labels'])
            labels = batch['labels'].to(self.settings.device)
            outputs = peft_model(**inputs)
            loss = cross_entropy(outputs.logits, labels)  # Calculate loss

            loss.backward()
            optimizer.step()

        self.is_training = False
        self.logger.info("Model training completed and took %f s.", elapsed_time)

        return samples_trained_on
