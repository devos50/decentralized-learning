from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.model import create_model

parameters = {
        "batch_size": 20,
        "target_participants": 100,
        "dataset": "cifar10_niid",
        "participants": ["a"],
        "learning_rate": 0.001,
        "momentum": 0,
    }

data_dir = "/Users/martijndevos/dfl-data"

mapping = Linear(1, 100)
s = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

print("Datasets prepared")

# Model
model = create_model(parameters["dataset"])
print(model)

print("Initial evaluation")
print(s.test(model))

for round in range(3):
        # Train
        trainer = ModelTrainer(data_dir, parameters, 0)
        trainer.train(model)
        print("Training done")
        print(s.test(model))
