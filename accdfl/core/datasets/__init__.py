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

    if settings.dataset in ["ag_news", "emotion"]:
        dataset = FederatedDataset(
            dataset=settings.dataset,
            partitioners={"train": partitioner},
            cache_dir="data/datasets",
        )
    elif settings.dataset == "newsgroups":
        dataset = FederatedDataset(
            dataset="SetFit/20_newsgroups",
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
