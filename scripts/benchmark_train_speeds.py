import asyncio
import time
from typing import Dict

import torch

from transformers import PreTrainedModel, DataCollatorWithPadding, ViTImageProcessor

from accdfl.core.datasets import create_global_dataset, group_texts, tokenize_dataset, tokenize_txt_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_adapters, create_base_model, create_tokenizer, serialize_adapter
from accdfl.core.session_settings import LearningSettings, SessionSettings


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def create_datasets_and_model(session_settings):
    # Create the global dataset
    dataset = create_global_dataset(session_settings)
    dataset._prepare_dataset()

    # Create the base model
    base_model: PreTrainedModel = create_base_model(session_settings.model, session_settings.dataset, dataset._dataset)

    # Create the adapters
    peft_config, peft_model, adapters, global_adapter = create_adapters(session_settings, base_model)
    peft_model.to(device)

    # Compute the size of the serialized adapter
    model_state_dict: Dict = peft_model.state_dict()
    global_adapter_dict: Dict = {}
    for k in global_adapter["keys"]:
        global_adapter_dict[k] = model_state_dict[k]

    # Create each of the datasets
    split_datasets = [dataset.load_partition(i, "train") for i in range(len(session_settings.participants))]

    # Tokenize the datasets (tonkenization, etc.)
    if session_settings.model == "google/vit-base-patch16-224":
        feature_extractor = ViTImageProcessor.from_pretrained(session_settings.model, cache_dir="data/models")

        def transform(batch):
            # batch["img"] is a list of PIL Images; batch["label"] is a list/array of ints
            out = feature_extractor(batch["img"], return_tensors="pt")
            # Make sure labels are a 1D LongTensor of length batch_size
            out["labels"] = torch.tensor(batch["label"], dtype=torch.long)
            return out

        for ind in range(len(split_datasets)):
            split_datasets[ind] = split_datasets[ind].with_transform(transform)
        test_dataset = dataset.load_split("test").with_transform(transform)
    elif session_settings.model == "gpt2":
        tokenizer = create_tokenizer(session_settings)

        def text_collate(examples):
            input_ids = torch.stack([torch.tensor(d["input_ids"]) for d in examples])
            labels = torch.stack([torch.tensor(d["labels"]) for d in examples])
            attention_mask = torch.stack([torch.tensor(d["attention_mask"]) for d in examples])
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        data_collator = text_collate

        # Tokenize and group
        for ind in range(len(split_datasets)):
            ds_tok = tokenize_txt_dataset(split_datasets[ind], tokenizer)
            split_datasets[ind] = ds_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)
        test_tok = tokenize_txt_dataset(dataset.load_split("test"), tokenizer)
        test_dataset = test_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)
    else:
        tokenizer = create_tokenizer(session_settings)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        for ind in range(len(split_datasets)):
            split_datasets[ind] = tokenize_dataset(split_datasets[ind], tokenizer)
        test_dataset = tokenize_dataset(dataset.load_split("test"), tokenizer).rename_column("label", "labels")

    return split_datasets, adapters, global_adapter, tokenizer, peft_model, data_collator


async def main():
    settings = SessionSettings(
        work_dir="data",
        dataset="wikitext",
        model="gpt2",
        learning=LearningSettings(batch_size=16, learning_rate=0.1, momentum=0.9, weight_decay=0, local_steps=10),
        participants=["a"],
        all_participants=["a"],
        target_participants=10,
        device=device,
    )

    split_datasets, adapters, global_adapter, tokenizer, peft_model, data_collator = create_datasets_and_model(settings)

    model_trainer = ModelTrainer(settings, 0)
    model_trainer.setup_dataset(split_datasets[0], tokenizer, data_collator)
    start_time = time.time()
    await model_trainer.train(peft_model)
    train_duration_per_local_step = (time.time() - start_time) / settings.learning.local_steps
    print(f"Training time per local step: {train_duration_per_local_step} seconds")


if __name__ == "__main__":
    asyncio.run(main())
