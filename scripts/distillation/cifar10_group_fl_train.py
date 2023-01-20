"""
Test to distill knowledge from a CIFAR10 pre-trained model to another one.
"""
import asyncio
import copy
import logging
import os
import time

import torch

from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.mappings import Linear
from accdfl.core.model_manager import ModelManager
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import LearningSettings, SessionSettings

NUM_ROUNDS = 100 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])
NUM_PEERS = 100 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])
GROUP_SIZE = 10 if "GROUP_SIZE" not in os.environ else int(os.environ["GROUP_SIZE"])

if NUM_PEERS % GROUP_SIZE != 0:
    raise RuntimeError("Invalid number of peers/group size ratio")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cifar10_group_fl_train")


async def run():
    learning_settings = LearningSettings(
        learning_rate=0.002 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
        momentum=0.9,
        batch_size=20,
    )

    settings = SessionSettings(
        dataset="cifar10",
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=NUM_PEERS,
    )

    data_path = os.path.join("data", "n_%d" % NUM_PEERS)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,group,peers,round,learning_rate,accuracy,loss\n")

    # Create the teacher models
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")
    model_manager = ModelManager(None, settings, 0)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    mapping = Linear(1, 1)
    cifar10_testset = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

    # Train the model using FL per group
    for group in range(NUM_PEERS // GROUP_SIZE):
        logger.info("Starting FL process for group %d", group)
        models = [create_model(settings.dataset) for n in range(10)]
        trainers = [ModelTrainer(data_dir, settings, GROUP_SIZE * group + n) for n in range(10)]
        highest_acc = 0

        for round_nr in range(NUM_ROUNDS):
            model_manager.reset_incoming_trained_models()
            for n in range(GROUP_SIZE):
                start_time = time.time()
                await trainers[n].train(models[n], device_name=device)
                logger.info("Training round %d for peer %d done - time: %f", round_nr + 1, n, time.time() - start_time)
                model_manager.process_incoming_trained_model(b"%d" % n, models[n])

            # Average the models and replace them
            avg_model = model_manager.aggregate_trained_models()

            acc, loss = cifar10_testset.test(avg_model, device_name=device)
            logger.info("Test accuracy: %f, loss: %f (group %d, round %d)" % (acc, loss, group, round_nr))
            with open(os.path.join(data_path, "accuracies.csv"), "a") as out_file:
                out_file.write("cifar10,%d,%d,%d,%f,%f,%f\n" % (group, NUM_PEERS, round_nr, learning_settings.learning_rate, acc, loss))
            if acc > highest_acc:
                torch.save(avg_model.state_dict(), os.path.join(data_path, "cifar10_%d.model" % group))
                highest_acc = acc

            # Replace the local models of peers in the group with the average model
            for n in range(GROUP_SIZE):
                models[n] = copy.deepcopy(avg_model)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
