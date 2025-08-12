from typing import Optional
import torch
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

import evaluate

from peft import PeftModel

from accdfl.core.session_settings import SessionSettings


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    """

    def __init__(self, settings: SessionSettings):
        self.settings: SessionSettings = settings
        self.test_dataset = None

    def setup_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer, data_collator: Optional[DataCollatorWithPadding]):
        self.test_dataset = dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def evaluate_classification_model(self, inference_model):
        metric = evaluate.load('accuracy')
        eval_dataloader = DataLoader(self.test_dataset, batch_size=512, collate_fn=self.data_collator)

        inference_model.to(self.settings.device)
        inference_model.eval()

        total_loss = 0.0
        n_examples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.settings.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = inference_model(**batch)
                logits = outputs.logits
                loss = outputs.loss

                labels = batch["labels"]
                preds = logits.argmax(dim=-1)

                # accuracy
                metric.add_batch(predictions=preds, references=labels)

                # loss: weight by batch size to get dataset mean
                bs = labels.size(0)
                total_loss += loss.item() * bs
                n_examples += bs

        acc = metric.compute()["accuracy"]
        mean_loss = total_loss / max(n_examples, 1)
        return {"accuracy": acc, "loss": mean_loss}

    def evaluate_accuracy(self, peft_model: PeftModel, adapter_to_test: str = "global"):
        peft_model.set_adapter(adapter_to_test)
        eval_res = self.evaluate_classification_model(peft_model)
        return eval_res['accuracy'], eval_res['loss']
