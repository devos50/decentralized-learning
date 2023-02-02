import os
import time

import torch

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


async def run(learning_settings: LearningSettings, dataset: str):
    num_rounds = 100 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])
    num_peers = 10 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])

    settings = SessionSettings(
        dataset=dataset,
        partitioner="iid" if "PARTITIONER" not in os.environ else os.environ["PARTITIONER"],
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=num_peers,
    )

    data_dir = os.path.join(os.environ["HOME"], "dfl-data")
    test_dataset = create_dataset(settings, 0, test_dir=data_dir)

    print("Datasets prepared")

    data_path = os.path.join("data", "%s_n_%d" % (dataset, num_peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    # Model
    models = [create_model(settings.dataset, architecture=settings.model) for n in range(num_peers)]
    trainers = [ModelTrainer(data_dir, settings, n) for n in range(num_peers)]

    for n in range(num_peers):
        highest_acc, lowest_loss = 0, 0
        for round in range(num_rounds):
            start_time = time.time()
            print("Starting training round %d for peer %d" % (round + 1, n))
            await trainers[n].train(models[n], device_name=device)
            print("Training round %d for peer %d done - time: %f" % (round + 1, n, time.time() - start_time))
            acc, loss = test_dataset.test(models[n], device_name=device)
            print("Accuracy: %f, loss: %f" % (acc, loss))

            # Save the model if it's better
            if acc > highest_acc:
                torch.save(models[n].state_dict(), os.path.join(data_path, "cifar10_%d.model" % n))
                highest_acc = acc
                lowest_loss = loss

        # Write the final accuracy
        with open(os.path.join(data_path, "accuracies.csv"), "a") as out_file:
            out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % (dataset, "standalone", n, num_peers, num_rounds, learning_settings.learning_rate, highest_acc, lowest_loss))
