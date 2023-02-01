"""
Train n standalone Spambase models for a specified number of rounds.
"""
import asyncio
import logging
import os
import time

import torch

from accdfl.core.datasets.spambase import Spambase
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings

NUM_ROUNDS = 10 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])
NUM_PEERS = 1 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])

logging.basicConfig(level=logging.INFO)

learning_settings = LearningSettings(
    learning_rate=1 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
    momentum=0,
    batch_size=32
)

settings = SessionSettings(
    dataset="spambase",
    partitioner="iid" if "PARTITIONER" not in os.environ else os.environ["PARTITIONER"],
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=NUM_PEERS,
)

data_dir = os.path.join(os.environ["HOME"], "dfl-data")

mapping = Linear(1, NUM_PEERS)
test_dataset = Spambase(0, 0, mapping, settings.partitioner, test_dir=data_dir)

print("Datasets prepared")

data_path = os.path.join("data", "n_%d" % NUM_PEERS)
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
    out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print("Device to train/determine accuracy: %s" % device)

# Model
models = [create_model(settings.dataset, architecture=settings.model) for n in range(NUM_PEERS)]
trainers = [ModelTrainer(data_dir, settings, n) for n in range(NUM_PEERS)]


async def run():
    for n in range(NUM_PEERS):
        highest_acc, lowest_loss = 0, 0
        for round in range(NUM_ROUNDS):
            start_time = time.time()
            print("Starting training round %d for peer %d" % (round + 1, n))
            await trainers[n].train(models[n], device_name=device)
            print("Training round %d for peer %d done - time: %f" % (round + 1, n, time.time() - start_time))
            acc, loss = test_dataset.test(models[n], device_name=device)
            print("Accuracy: %f, loss: %f" % (acc, loss))

            # Save the model if it's better
            if acc > highest_acc:
                torch.save(models[n].state_dict(), os.path.join(data_path, "spambase_%d.model" % n))
                highest_acc = acc
                lowest_loss = loss

        # Write the final accuracy
        with open(os.path.join(data_path, "accuracies.csv"), "a") as out_file:
            out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % ("cifar10", "standalone", n, NUM_PEERS, NUM_ROUNDS, learning_settings.learning_rate, highest_acc, lowest_loss))


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
