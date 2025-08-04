import torch
import pytest

from accdfl.core.datasets import create_global_dataset
from accdfl.core.models import create_adapters, create_base_model, create_tokenizer
from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import SessionSettings, LearningSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="ag_news",
        model="roberta-base",
        learning=LearningSettings(batch_size=20, learning_rate=0.002, momentum=0.9, weight_decay=0, local_steps=5),
        participants=["a", "b"],
        all_participants=["a", "b"],
        target_participants=2
    )


@pytest.fixture
def model_and_adapters(settings):
    dataset = create_global_dataset(settings)
    base_model = create_base_model(settings.model, settings.dataset, dataset)
    tokenizer = create_tokenizer(settings)
    peft_config, peft_model, adapters, global_adapter = create_adapters(settings, base_model)
    return dataset, tokenizer, peft_config, peft_model, adapters, global_adapter


def test_adapter_aggregation_and_adoptation(settings, model_and_adapters):
    dataset, tokenizer, peft_config, peft_model, adapters, global_adapter = model_and_adapters
    model_manager = ModelManager(peft_model, settings, 0)
    model_manager.adapter = adapters[0]
    assert len(adapters) == 2, "There should be two adapters for two participants"

    weight_name = "base_model.model.roberta.encoder.layer.3.attention.self.query.lora_A.weight"
    weights = global_adapter[weight_name]
    global_weights_before = weights.clone().detach()

    # Try aggregating adapter 1 and 2
    model_manager.process_incoming_trained_adapter(b'peer1', adapters[0])
    model_manager.process_incoming_trained_adapter(b'peer2', adapters[1])

    model_manager.aggregate_trained_adapters()

    # Check if the global adapter has changed
    global_weights_after = global_adapter[weight_name].clone().detach()
    assert not torch.allclose(global_weights_before, global_weights_after), "Global adapter weights should change after aggregation"

    client1_weights_before = adapters[0][weight_name].clone().detach()
    client2_weights_before = adapters[1][weight_name].clone().detach()
    model_manager.adopt_adapter(global_adapter)

    # Check if the client adapter has changed
    client1_weights_after = adapters[0][weight_name].clone().detach()
    assert not torch.allclose(client1_weights_before, client1_weights_after), "Client 1 adapter weights should change after adopting the global adapter"

    client2_weights_after = adapters[1][weight_name].clone().detach()
    assert torch.allclose(client2_weights_before, client2_weights_after), "Client 2 adapter weights should not change after adopting the global adapter"
