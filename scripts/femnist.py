import os

from accdfl.core.datasets.Femnist import Femnist, CNN
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer

parameters = {
        "batch_size": 20,
        "target_participants": 100,
        "dataset": "femnist",
        "participants": ["a"],
        "learning_rate": 0.001,
        "momentum": 0,
    }

data_dir = "/Users/martijndevos/leaf/femnist"
train_dir = os.path.join(data_dir, "per_user_data/train")
test_dir = os.path.join(data_dir, "data/test")

mapping = Linear(1, 100)
s = Femnist(0, 0, mapping, train_dir=train_dir, test_dir=test_dir)

print("Datasets prepared")

# Model
model = CNN()
print(model)

print("Initial evaluation")
#print(s.test(model))

for round in range(3):
        # Train
        trainer = ModelTrainer(data_dir, parameters, 0)
        trainer.train(model)
        print("Training done")
        print(s.test(model))
