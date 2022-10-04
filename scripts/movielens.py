import logging
import os
import time

from accdfl.core.datasets.MovieLens import MovieLens, MatrixFactorization
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.session_settings import LearningSettings, SessionSettings

NUM_ROUNDS = 20

logging.basicConfig(level=logging.INFO)

learning_settings = LearningSettings(
    learning_rate=0.25,
    momentum=0,
    batch_size=20
)

settings = SessionSettings(
    dataset="movielens",
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=1,
)

data_dir = os.path.join(os.environ["HOME"], "leaf", "movielens")

mapping = Linear(1, 100)
s = MovieLens(1, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print("Datasets prepared")

train_dataset = s.get_trainset()
test_dataset = s.get_testset()

print("Train dataset items: %d" % len(train_dataset.dataset))
print("Test dataset items: %d" % len(test_dataset.dataset))

# Model
model = MatrixFactorization()
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
