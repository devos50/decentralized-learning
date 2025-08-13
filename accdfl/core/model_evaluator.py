import math
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
    
    def evaluate_lm(self, eval_model):
        """Compute token-avg NLL and perplexity on a pre-tokenized dataset."""
        eval_model.to(self.settings.device)
        eval_model.eval()

        # If you used fixed-size blocks with no padding, default collator is fine.
        # Otherwise, make sure your collator returns tensors incl. 'labels'.
        eval_dataloader = DataLoader(
            self.test_dataset,
            batch_size=256,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # move to device
                batch = {k: v.to(self.settings.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # forward with labels so model computes CE (ignore_index=-100)
                outputs = eval_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss  # scalar CE averaged over valid labels in the batch

                # weight by the number of valid (non -100) label tokens
                valid = (batch["labels"] != -100).sum().item()
                total_nll += loss.item() * valid
                total_tokens += valid

        mean_nll = total_nll / max(total_tokens, 1)
        ppl = math.exp(mean_nll)
        print(ppl)
        return {"accuracy": 0.0, "loss": mean_nll, "perplexity": ppl}

    def evaluate_accuracy(self, peft_model: PeftModel, adapter_to_test: str = "global"):
        peft_model.set_adapter(adapter_to_test)
        if self.settings.model == "gpt2":
            eval_res = self.evaluate_lm(peft_model)
        else:
            eval_res = self.evaluate_classification_model(peft_model)
        return eval_res['accuracy'], eval_res['loss']
