import logging
import os
import time

from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.model import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings

NUM_ROUNDS = 3

logging.basicConfig(level=logging.DEBUG)

learning_settings = LearningSettings(
    learning_rate=0.001,
    momentum=0,
    batch_size=200
)

settings = SessionSettings(
    dataset="cifar10_niid",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=1,
)

data_dir = os.path.join(os.environ["HOME"], "dfl-data")

mapping = Linear(1, 100)
s = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print("Datasets prepared")

# Model
model = create_model(settings.dataset)
print(model)

print("Initial evaluation")
print(s.test(model))

for round in range(NUM_ROUNDS):
    start_time = time.time()
    print("Starting training round %d" % (round + 1))
    trainer = ModelTrainer(data_dir, settings, 0)
    trainer.train(model)
    print("Training round %d done - time: %f" % (round + 1, time.time() - start_time))
    print(s.test(model))
