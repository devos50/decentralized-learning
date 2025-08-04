import os

import pytest

from accdfl.core.datasets import create_global_dataset, tokenize_dataset
from accdfl.core.models import create_adapters, create_base_model, create_tokenizer
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings, LearningSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="ag_news",
        model="roberta-base",
        learning=LearningSettings(batch_size=20, learning_rate=0.002, momentum=0.9, weight_decay=0, local_steps=5),
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
def model_evaluator(settings):
    return ModelEvaluator(settings)


def test_evaluate(settings, model_and_adapters, model_evaluator):
    dataset, tokenizer, _, peft_model, _, _ = model_and_adapters
    test_dataset = dataset["test"]
    test_dataset = test_dataset.select(range(1))
    test_dataset = tokenize_dataset(test_dataset, tokenizer)
    model_evaluator.setup_dataset(test_dataset, tokenizer)
    res = model_evaluator.evaluate_accuracy(peft_model)
    assert res is not None
    assert len(res) == 2
