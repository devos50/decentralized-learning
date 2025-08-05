import datasets
from datasets import ClassLabel, Dataset

from accdfl.core.session_settings import SessionSettings


def create_global_dataset(settings: SessionSettings) -> Dataset:
    if settings.dataset in ["ag_news", "emotion"]:
        dataset = datasets.load_dataset(settings.dataset, cache_dir="data/datasets")
        return dataset
    elif settings.dataset == "newsgroups":
        dataset = datasets.load_dataset("SetFit/20_newsgroups", cache_dir="data/datasets")
        # We need to do some small transformations
        unique_classes = sorted(set(dataset['train']['label']))
        label_feature = ClassLabel(names=unique_classes)
        dataset = dataset.cast_column('label', label_feature)
        dataset = dataset.remove_columns('label_text')
        return dataset
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)


def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_dataset = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True,  remove_columns=["text"])
    return processed_dataset
