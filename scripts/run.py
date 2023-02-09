import argparse
import os
import time

import torch

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


def get_args(default_lr: float, default_momentum: float = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--momentum', type=float, default=default_momentum)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--peers', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()


async def run(args, dataset: str):
    learning_settings = LearningSettings(
        learning_rate=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size
    )

    settings = SessionSettings(
        dataset=dataset,
        partitioner=args.partitioner,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=args.peers,
    )

    test_dataset = create_dataset(settings, 0, test_dir=args.data_dir)

    print("Datasets prepared")

    data_path = os.path.join("data", "%s_n_%d" % (dataset, args.peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    # Model
    models = [create_model(settings.dataset, architecture=settings.model) for _ in range(args.peers)]
    print("Created %d models of type %s..." % (len(models), models[0].__class__.__name__))
    trainers = [ModelTrainer(args.data_dir, settings, n) for n in range(args.peers)]

    for n in range(args.peers):
        highest_acc, lowest_loss = 0, 0
        for round in range(args.rounds):
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
            out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % (dataset, "standalone", n, args.peers, args.rounds, learning_settings.learning_rate, highest_acc, lowest_loss))
