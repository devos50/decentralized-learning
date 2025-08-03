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
        self.metric = evaluate.load('accuracy')

    def setup_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer):
        self.test_dataset = dataset
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

    def evaluate_classification_model(self, inference_model):
        TASK = "txt_classification"
        eval_dataloader = DataLoader(self.test_dataset.rename_column("label", "labels") if TASK != "img_classification" else self.test_dataset, batch_size=512, collate_fn=self.data_collator)

        inference_model.to(self.settings.device)
        inference_model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch = {key: val.to(self.settings.device) for key, val in batch.items() if isinstance(val, torch.Tensor)}
            with torch.no_grad():
                outputs = inference_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            self.metric.add_batch(
                predictions=predictions,
                references=references,
            )

        return self.metric.compute()

    def evaluate_accuracy(self, peft_model: PeftModel):
        peft_model.set_adapter("global")
        eval_res = self.evaluate_classification_model(peft_model)
        return eval_res['accuracy'], eval_res['loss']
