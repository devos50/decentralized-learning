import logging
import pickle
from typing import Dict, List, Tuple
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, ViTForImageClassification

from accdfl.core.session_settings import SessionSettings


logger = logging.getLogger(__name__)


def serialize_adapter(adapter: Dict) -> bytes:
    """
    Serialize a PEFT adapter to bytes.
    """
    return pickle.dumps(adapter)


def unserialize_adapter(serialized_adapter: bytes):
    """
    Unserialize a PEFT adapter from bytes.
    """
    return pickle.loads(serialized_adapter)


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
    elif dataset_name == "cifar10":
        return ViTForImageClassification.from_pretrained(base_model_name, num_labels=10, ignore_mismatched_sizes=True, cache_dir="data/models")
    else:
        raise RuntimeError("Unknown dataset %s" % dataset_name)


def create_tokenizer(session_settings: SessionSettings) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(session_settings.model, use_fast=True)


def create_adapters(session_settings: SessionSettings, base_model: PreTrainedModel) -> Tuple[LoraConfig, PeftModel, List[Dict], Dict]:
    if session_settings.model == "roberta-base":
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    elif session_settings.model == "google/vit-base-patch16-224":
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["attention.query", "attention.key"])
    # TODO these parameters should be configurable
    peft_model = get_peft_model(base_model, peft_config, adapter_name="global")

    # Create adapters for each user
    for adapter_name in ["client_%d" % i for i in range(len(session_settings.participants))]:
        peft_model.add_adapter(adapter_name, peft_config)

    # Create the adapter dictionaries, we get the keys directly from the PEFT model
    adapters = []
    model_state_dict: Dict = peft_model.state_dict()
    for adapter_name in ["client_%d" % i for i in range(len(session_settings.participants))]:
        adapter: Dict = {"name": adapter_name, "keys": []}
        for key in model_state_dict.keys():
            if f".{adapter_name}." in key:
                adapter["keys"].append(key)
        adapters.append(adapter)

    # Add the global adapter
    global_adapter = {"name": "global", "keys": []}
    for key in model_state_dict.keys():
        if ".global." in key:
            global_adapter["keys"].append(key)

    return peft_config, peft_model, adapters, global_adapter
