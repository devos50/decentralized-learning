import argparse
import logging
import os
import random
import time
from typing import List

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


def get_args(default_lr: float, default_momentum: float = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=default_momentum)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--peers', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--check-interval', type=int, default=1)
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()


class DatasetWithIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


logger = logging.getLogger('Distillation')


async def run(args, dataset: str):
    learning_settings = LearningSettings(
        learning_rate=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        beta=args.beta
    )

    settings = SessionSettings(
        dataset=dataset,
        partitioner=args.partitioner,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=args.peers,
        model=args.model,
    )

    dist_settings = SessionSettings(
        dataset="cifar100",
        partitioner=args.partitioner,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
        model=args.model,
    )

    test_dataset = create_dataset(settings, 0, test_dir=args.data_dir)
    proxy_dataset = create_dataset(dist_settings, 0, train_dir=args.data_dir)

    print("Datasets prepared")

    data_path = os.path.join("data", "%s_n_%d" % (dataset, args.peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,beta,learning_rate,accuracy,loss\n")

    with open(os.path.join(data_path, "train.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,local_loss,distill_loss\n")

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    # Model
    models = [create_model(settings.dataset, architecture=settings.model) for _ in range(args.peers)]
    print("Created %d models of type %s..." % (len(models), models[0].__class__.__name__))
    trainers = [ModelTrainer(args.data_dir, settings, n) for n in range(args.peers)]

    highest_accs, lowest_losses = [0] * args.peers, [0] * args.peers
    ordered_proxy_trainset = proxy_dataset.get_trainset(batch_size=settings.learning.batch_size, shuffle=False)
    for round in range(1, args.rounds + 1):
        print("Starting communication round %d" % round)

        # Determine outputs of the teacher model on the public training data
        outputs: List[List[Tensor]] = []
        outputs_indices: List[List[int]] = []
        proxy_trainset = DataLoader(DatasetWithIndex(ordered_proxy_trainset.dataset), batch_size=settings.learning.batch_size, shuffle=True)
        train_set_it = iter(proxy_trainset)

        try:
            data, _, indices = next(train_set_it)
        except StopIteration:
            proxy_trainset = DataLoader(DatasetWithIndex(ordered_proxy_trainset.dataset), batch_size=settings.learning.batch_size, shuffle=True)
            train_set_it = iter(proxy_trainset)
            data, _, indices = next(train_set_it)
        data = Variable(data.to(device))

        for n in range(args.peers):
            models[n].to(device)
            teacher_outputs = []
            teacher_indices = []

            # Do one batch of inferences
            out = torch.softmax(models[n].forward(data).detach(), dim=1)
            teacher_outputs += out
            teacher_indices += indices

            outputs.append(teacher_outputs)
            outputs_indices.append(teacher_indices)

        # # Aggregate the predicted outputs
        # print("Aggregating predictions...")
        # aggregated_predictions = []
        # for sample_ind in range(len(outputs[0])):
        #     predictions = [outputs[n][sample_ind] for n in range(args.peers)]
        #     aggregated_predictions.append(torch.mean(torch.stack(predictions), dim=0))

        for n in range(args.peers):
            start_time = time.time()

            # Choose which peer we're going to distill from
            possibilities = [i for i in range(settings.target_participants) if i != n]
            if not possibilities:
                peer_to_distill_from = n  # Distill from self
            else:
                peer_to_distill_from = random.choice(possibilities)
            logger.debug("Peer %d distilling from peer %d", n, peer_to_distill_from)

            predictions = outputs[peer_to_distill_from], outputs_indices[peer_to_distill_from]
            _, local_loss, distill_loss = await trainers[n].train(models[n], device_name=device, proxy_dataset=ordered_proxy_trainset, predictions=predictions)
            logger.debug("Training round %d for peer %d done - time: %f", round, n, time.time() - start_time)

            if round % args.check_interval == 0:
                with open(os.path.join(data_path, "train.csv"), "a") as out_file:
                    out_file.write("%s,%s,%d,%d,%d,%f,%f\n" % (dataset, "distill", n, args.peers, round, local_loss, distill_loss))

                acc, loss = test_dataset.test(models[n], device_name=device)
                print("Accuracy: %f, loss: %f" % (acc, loss))

                # Save the model if it's better
                if acc > highest_accs[n]:
                    torch.save(models[n].state_dict(), os.path.join(data_path, "cifar10_%d.model" % n))
                    highest_accs[n] = acc
                    lowest_losses[n] = loss

                # Write the final accuracy
                with open(os.path.join(data_path, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,%s,%d,%d,%d,%d,%f,%f,%f\n" % (dataset, "distill", n, args.peers, round, learning_settings.beta, learning_settings.learning_rate, acc, loss))
