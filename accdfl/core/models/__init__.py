import pickle
from typing import Dict
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from accdfl.core.session_settings import SessionSettings


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


def create_base_model(dataset_name: str, dataset: Dataset) -> PreTrainedModel:
    if dataset_name == "ag_news":
        # Extract the number of classess and their names
        num_labels = dataset['train'].features['label'].num_classes
        class_names = dataset["train"].features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")
        
        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return AutoModelForSequenceClassification.from_pretrained("roberta-base", id2label=id2label, cache_dir="data/models")
    else:
        raise RuntimeError("Unknown dataset %s" % dataset_name)


def create_adapters(session_settings: SessionSettings, base_model: PreTrainedModel) -> PeftModel:
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    # TODO these parameters should be configurable
    peft_model = get_peft_model(base_model, peft_config)

    # Create adapters for each user
    for adapter_name in ["client_%d" % i for i in range(len(session_settings.participants))]:
        if adapter_name not in peft_model.peft_config:
            peft_model.add_adapter(adapter_name, peft_config)
        peft_model.set_adapter(adapter_name)

    # Create a global adapter (for the aggregation)
    if "global" not in peft_model.peft_config:
        peft_model.add_adapter("global", peft_config)

    return peft_model
