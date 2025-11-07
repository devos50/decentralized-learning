import logging
from datasets import Dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, ViTForImageClassification

from accdfl.core.session_settings import SessionSettings


logger = logging.getLogger(__name__)


def serialize_chunk(chunk) -> bytes:
    return chunk.numpy().astype(np.float32).tobytes()


def create_base_model(base_model_name: str, dataset_name: str, dataset: Dataset) -> PreTrainedModel:
    if dataset_name in ["ag_news", "emotion", "newsgroups"]:
        # Extract the number of classess and their names
        num_labels = dataset['train'].features['label'].num_classes
        class_names = dataset["train"].features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")
        
        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return AutoModelForSequenceClassification.from_pretrained(base_model_name, id2label=id2label, cache_dir="data/models")
    elif dataset_name == "wikitext":
        return AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir="data/models")
    elif dataset_name == "cifar10":
        return ViTForImageClassification.from_pretrained(base_model_name, num_labels=10, ignore_mismatched_sizes=True, cache_dir="data/models")
    elif dataset_name == "food101":
        return ViTForImageClassification.from_pretrained(base_model_name, num_labels=101, ignore_mismatched_sizes=True, cache_dir="data/models")
    else:
        raise RuntimeError("Unknown dataset %s" % dataset_name)


def create_tokenizer(session_settings: SessionSettings) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(session_settings.model, use_fast=True)
