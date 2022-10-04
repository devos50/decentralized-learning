import logging
import os
import time

from accdfl.core.datasets.Celeba import Celeba, CNN
from accdfl.core.mappings import Linear
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
    dataset="celeba",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=100,
)

data_dir = os.path.join(os.environ["HOME"], "leaf", "celeba")
train_dir = os.path.join(data_dir, "per_user_data/train")
test_dir = os.path.join(data_dir, "data/test")

mapping = Linear(1, 1000)
s = Celeba(0, 0, mapping, train_dir=train_dir, test_dir=test_dir, images_dir=os.path.join(data_dir, "data", "raw", "img_align_celeba"))

print("Datasets prepared")

# Model
model = CNN()
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
