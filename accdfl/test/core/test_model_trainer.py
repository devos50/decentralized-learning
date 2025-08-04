import torch
import pytest

from accdfl.core.datasets import create_global_dataset, tokenize_dataset
from accdfl.core.models import create_adapters, create_base_model, create_tokenizer
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import SessionSettings, LearningSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="ag_news",
        model="roberta-base",
        learning=LearningSettings(batch_size=5, learning_rate=0.1, momentum=0.9, weight_decay=0, local_steps=1),
        participants=["a"],
        all_participants=["a"],
        target_participants=100
    )


@pytest.fixture
def model_and_adapters(settings):
    dataset = create_global_dataset(settings)
    base_model = create_base_model(settings.model, settings.dataset, dataset)
    tokenizer = create_tokenizer(settings)
    peft_config, peft_model, adapters, global_adapter = create_adapters(settings, base_model)
    return dataset, tokenizer, peft_config, peft_model, adapters, global_adapter


@pytest.fixture
def model_trainer(settings) -> ModelTrainer:
    return ModelTrainer(settings, 0)


@pytest.mark.asyncio
async def test_train(settings, model_and_adapters, model_trainer):
    dataset, tokenizer, _, peft_model, adapters, _ = model_and_adapters

    client_adapter = adapters[0]
    weight_name = "base_model.model.roberta.encoder.layer.3.attention.self.query.lora_A.weight"
    weights = client_adapter[weight_name]
    weights_before = weights.clone().detach()

    train_dataset = dataset["train"]
    train_dataset = train_dataset.select(range(5))
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    model_trainer.setup_dataset(train_dataset, tokenizer)
    await model_trainer.train(peft_model)
    
    weights_after = client_adapter[weight_name].detach()
    assert not torch.allclose(weights_before, weights_after), "Weights did not change after training!"
