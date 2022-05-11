import os

from accdfl.core.datasets.MovieLens import MovieLens, MatrixFactorization
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer

parameters = {
        "batch_size": 20,
        "target_participants": 1,
        "dataset": "movielens",
        "participants": ["a"],
        "learning_rate": 0.25,
        "momentum": 0,
    }

data_dir = "/Users/martijndevos/leaf/movielens"

mapping = Linear(1, 1)
s = MovieLens(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print("Datasets prepared")

# Model
model = MatrixFactorization()
print(model)

print("Initial evaluation")
print(s.test(model))

for round in range(20):
        # Train
        trainer = ModelTrainer(data_dir, parameters, 0)
        trainer.train(model)
        print("Training done")
        print(s.test(model))
