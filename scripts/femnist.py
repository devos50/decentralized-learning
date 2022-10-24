import logging
import os
import time

from accdfl.core.datasets.Femnist import Femnist
from accdfl.core.mappings import Linear
from accdfl.core.models import create_model
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import LearningSettings, SessionSettings

NUM_ROUNDS = 3

logging.basicConfig(level=logging.INFO)

learning_settings = LearningSettings(
    learning_rate=0.001,
    momentum=0,
    batch_size=20
)

settings = SessionSettings(
    dataset="femnist",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=100,
)

data_dir = os.path.join(os.environ["HOME"], "leaf", "femnist")
train_dir = os.path.join(data_dir, "per_user_data/train")
test_dir = os.path.join(data_dir, "data/test")

mapping = Linear(1, 100)
s = Femnist(0, 0, mapping, train_dir=train_dir, test_dir=test_dir)

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
