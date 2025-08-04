import datasets
from datasets import Dataset

from accdfl.core.session_settings import SessionSettings


def create_global_dataset(settings: SessionSettings) -> Dataset:
    if settings.dataset == "ag_news":
        dataset = datasets.load_dataset(settings.dataset, cache_dir="data/datasets")
        return dataset
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)


def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_dataset = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True,  remove_columns=["text"])
    return processed_dataset
