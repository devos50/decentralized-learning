import datasets
from datasets import Dataset

from accdfl.core.session_settings import SessionSettings


def create_global_dataset(settings: SessionSettings) -> Dataset:
    if settings.dataset == "ag_news":
        dataset = datasets.load_dataset(settings.dataset, cache_dir="data/datasets")
        return dataset
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
