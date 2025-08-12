import datasets
from datasets import ClassLabel, Dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from accdfl.core.session_settings import SessionSettings


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

    if settings.dataset in ["ag_news", "emotion", "cifar10"]:
        dataset = FederatedDataset(
            dataset=hf_dataset_name,
            partitioners={"train": partitioner},
            cache_dir="data/datasets",
        )
    elif settings.dataset == "newsgroups":
        dataset = FederatedDataset(
            dataset=hf_dataset_name,
            partitioners={"train": partitioner},
            cache_dir="data/datasets",
        )
        # We need to do some small transformations
        unique_classes = sorted(set(dataset['train']['label']))
        label_feature = ClassLabel(names=unique_classes)
        dataset = dataset.cast_column('label', label_feature)
        dataset = dataset.remove_columns('label_text')
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
    
    return dataset


def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_dataset = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True,  remove_columns=["text"])
    return processed_dataset
