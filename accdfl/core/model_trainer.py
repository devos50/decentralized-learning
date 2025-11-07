from asyncio import CancelledError, get_event_loop, sleep
import copy
import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedModel

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

    def setup_dataset(self, dataset: Dataset, tokenizer: Optional[AutoTokenizer], data_collator: Optional[DataCollatorWithPadding]):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    async def train(self, base_model: PreTrainedModel) -> Tuple[list, int]:
        """
        Train the model on a batch. Return an integer that indicates how many samples were trained on.
        """
        if self.settings.bypass_training:
            self.logger.info("Bypassing training as per settings.")
            return 0

        self.is_training = True
        samples_trained_on = 0
        base_model.to(self.settings.device)
        inner_model = copy.deepcopy(base_model).to(self.settings.device)
        inner_optimizer = torch.optim.AdamW(inner_model.parameters(), lr=self.settings.learning.client_learning_rate)

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

        inner_model.train()
        train_dataloader = DataLoader(self.dataset, batch_size=self.settings.learning.batch_size, shuffle=True, collate_fn=self.data_collator)
        train_set_it = iter(train_dataloader)

        for local_step in range(self.settings.learning.local_steps):
            try:
                batch = next(train_set_it)
            except StopIteration:
                # Restart the iterator if we run out of data
                train_dataloader = DataLoader(self.dataset, batch_size=self.settings.learning.batch_size, shuffle=True, collate_fn=self.data_collator)
                train_set_it = iter(train_dataloader)
                batch = next(train_set_it)

            inner_optimizer.zero_grad()

            is_lm = (self.settings.model in ["gpt2"])
            if is_lm:
                input_ids = batch["input_ids"].to(self.settings.device)
                labels = batch["labels"].to(self.settings.device)
                masks = batch["attention_mask"].to(self.settings.device)
                outputs = inner_model(input_ids=input_ids, labels=labels, attention_mask=masks)
                loss = outputs.loss
            else:
                inputs = {k: v.to(self.settings.device) for k, v in batch.items() if k != 'labels'}
                samples_trained_on += len(batch['labels'])
                labels = batch['labels'].to(self.settings.device)
                outputs = inner_model(**inputs)
                loss = cross_entropy(outputs.logits, labels)
            
            loss.backward()
            inner_optimizer.step()

        self.is_training = False
        self.logger.info("Model training completed and took %f s.", elapsed_time)

        # 2) Delta = θ_inner - θ_base (per-parameter list)
        with torch.no_grad():
            gradients = [p_i.detach().clone() - p_b.detach().clone() for p_i, p_b in zip(inner_model.parameters(), base_model.parameters())]

        return gradients, samples_trained_on

    def apply_outer_optimizer(self, base_model, gradients, outer_opt):
        with torch.no_grad():
            for p, d in zip(base_model.parameters(), gradients):
                p.grad = d.to(p.device).detach().clone()
        outer_opt.opt.step()
        outer_opt.opt.zero_grad(set_to_none=True)
