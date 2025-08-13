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
    
    def compute_perplexity(self, eval_model, encodings):
        max_length = 512 # eval_model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.settings.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            eval_model.eval()
        
            with torch.no_grad():
                outputs = eval_model(input_ids, labels=target_ids)
        
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss
        
            nlls.append(neg_log_likelihood)
        
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        return torch.exp(torch.stack(nlls).mean())

    def evaluate_accuracy(self, peft_model: PeftModel, adapter_to_test: str = "global"):
        peft_model.set_adapter(adapter_to_test)
        if self.settings.model == "gpt2":
            encodings = self.tokenizer("\n\n".join(self.test_dataset["text"]), return_tensors="pt")
            eval_res = self.compute_perplexity(peft_model, encodings)
        else:
            eval_res = self.evaluate_classification_model(peft_model)
        return eval_res['accuracy'], eval_res['loss']
