import datasets as hfds
from datasets import ClassLabel, Dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from accdfl.core.session_settings import SessionSettings


def _newsgroups_preproc(ds_dict: hfds.DatasetDict) -> hfds.DatasetDict:
    # Build id -> name mapping using the *unpartitioned* train split
    train = ds_dict["train"]
    id2name = {}
    for i, t in zip(train["label"], train["label_text"]):
        if i not in id2name:
            id2name[i] = t
    names = [id2name[i] for i in range(max(id2name) + 1)]
    label_feature = ClassLabel(names=names)

    fixed = {}
    for split, ds in ds_dict.items():
        if "label" in ds.column_names:
            ds = ds.cast_column("label", label_feature)
        if "label_text" in ds.column_names:
            ds = ds.remove_columns(["label_text"])
        fixed[split] = ds
    return hfds.DatasetDict(fixed)


def create_global_dataset(settings: SessionSettings) -> FederatedDataset:
    # Create the partitioner
    if settings.partitioner == "uniform":
        partitioner = IidPartitioner(num_partitions=len(settings.participants))
    elif settings.partitioner == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=len(settings.participants),
            partition_by="label",
            alpha=settings.alpha,
            min_partition_size=10,
            shuffle=True,
            seed=42,
        )

    hf_dataset_name = settings.dataset
    if settings.dataset == "newsgroups":
        hf_dataset_name = "SetFit/20_newsgroups"
    elif settings.dataset == "cifar10":
        hf_dataset_name = "uoft-cs/cifar10"
    elif settings.dataset == "food101":
        hf_dataset_name = "ethz/food101"

    if settings.dataset in ["ag_news", "emotion", "cifar10", "food101"]:
        dataset = FederatedDataset(
            dataset=hf_dataset_name,
            partitioners={"train": partitioner},
            cache_dir="data/datasets",
        )
    elif settings.dataset == "wikitext":
        dataset = FederatedDataset(
            dataset=hf_dataset_name,
            subset="wikitext-2-raw-v1",
            partitioners={"train": partitioner},
            cache_dir="data/datasets",
        )
    elif settings.dataset == "newsgroups":
        dataset = FederatedDataset(
            dataset=hf_dataset_name,
            partitioners={"train": partitioner},
            preprocessor=_newsgroups_preproc,
            cache_dir="data/datasets",
        )
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
    
    return dataset


def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_dataset = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True,  remove_columns=["text"])
    return processed_dataset


def preprocess_txt(examples, tokenizer):
        return tokenizer(examples["text"])

def tokenize_txt_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_dataset = dataset.map(preprocess_txt, fn_kwargs={"tokenizer": tokenizer}, batched=True,  remove_columns=["text"])
    return processed_dataset


def group_texts(examples):
    block_size = 128

    # Concatenate within this batch
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}  # keys: input_ids, attention_mask
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    # For causal LM, labels are a copy of input_ids (the model handles the shift internally)
    result["labels"] = result["input_ids"].copy()
    return result
