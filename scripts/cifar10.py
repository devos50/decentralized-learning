import asyncio
import logging
import os
import time

import torch

from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings

NUM_ROUNDS = 30

logging.basicConfig(level=logging.DEBUG)

learning_settings = LearningSettings(
    learning_rate=0.001,
    momentum=0.9,
    batch_size=200
)

settings = SessionSettings(
    dataset="cifar10",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=1,
)

data_dir = os.path.join(os.environ["HOME"], "dfl-data")

mapping = Linear(1, 1)
s = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print("Datasets prepared")

device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print("Device to train/determine accuracy: %s" % device)

# Model
model = create_model(settings.dataset)
print(model)

print("Initial evaluation")
print(s.test(model, device_name=device))

async def run():
    for round in range(NUM_ROUNDS):
        start_time = time.time()
        print("Starting training round %d" % (round + 1))
        trainer = ModelTrainer(data_dir, settings, 0)
        await trainer.train(model, device_name=device)
        print("Training round %d done - time: %f" % (round + 1, time.time() - start_time))
        print(s.test(model, device_name=device))

        # Save the model
        torch.save(model.state_dict(), "cifar10.model")


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
