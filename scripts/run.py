import argparse
import logging
import os
import stat
import subprocess
import sys
import time
from typing import List

import torch

from accdfl.core.datasets import create_dataset
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


logger = logging.getLogger("standalone-trainer")


def get_args(default_lr: float, default_momentum: float = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--momentum', type=float, default=default_momentum)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--max-peers-to-train', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--acc-check-interval', type=int, default=1)
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--train-method', type=str, default="local")
    parser.add_argument('--das-subprocess-jobs', type=int, default=1)
    parser.add_argument('--das-peers-per-subprocess', type=int, default=10)
    parser.add_argument('--das-peers-in-this-subprocess', type=str, default="")
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()


async def run(args, dataset: str):
    learning_settings = LearningSettings(
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    settings = SessionSettings(
        dataset=dataset,
        partitioner=args.partitioner,
        alpha=args.alpha,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=args.peers,
        model=args.model,
    )

    data_path = os.path.join("data", "%s_n_%d" % (dataset, args.peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if args.max_peers_to_train == 0:
        args.max_peers_to_train = args.peers

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

    if args.train_method == "local" or (args.train_method == "das" and args.das_peers_in_this_subprocess):
        await train_local(args, dataset, settings, data_path)
    elif args.train_method == "das" and not args.das_peers_in_this_subprocess:
        # This is the coordinating process
        await train_das(args, data_path)


async def train_local(args, dataset: str, settings: SessionSettings, data_path: str):
    test_dataset = create_dataset(settings, 0, test_dir=args.data_dir)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    # Determine which peers we should train for
    peers: List[int] = range(args.peers) if args.train_method == "local" else [int(n) for n in args.das_peers_in_this_subprocess.split(",")]

    for n in peers:
        model = create_model(settings.dataset, architecture=settings.model)
        trainer = ModelTrainer(args.data_dir, settings, n)
        highest_acc = 0
        for round in range(1, args.rounds + 1):
            start_time = time.time()
            print("Starting training round %d for peer %d" % (round, n))
            await trainer.train(model, device_name=device)
            print("Training round %d for peer %d done - time: %f" % (round, n, time.time() - start_time))

            if round % args.acc_check_interval == 0:
                acc, loss = test_dataset.test(model, device_name=device)
                print("Accuracy: %f, loss: %f" % (acc, loss))

                # Save the model if it's better
                if acc > highest_acc:
                    torch.save(model.state_dict(), os.path.join(data_path, "cifar10_%d.model" % n))
                    highest_acc = acc

                acc_file_name = "accuracies.csv" if args.train_method == "local" else "accuracies_%d.csv" % n
                with open(os.path.join(data_path, acc_file_name), "a") as out_file:
                    out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % (dataset, "standalone", n, args.peers, round, settings.learning.learning_rate, acc, loss))


async def train_das(args, data_path: str):
    """
    Test the accuracy of all models in the model manager by spawning different DAS jobs.
    """

    # Divide the clients over the DAS nodes
    client_queue = list(range(args.max_peers_to_train))
    while client_queue:
        logger.info("Scheduling new batch on DAS nodes - %d clients left", len(client_queue))

        processes = []
        for job_ind in range(args.das_subprocess_jobs):
            if not client_queue:
                continue

            clients_on_this_node = []
            while client_queue and len(clients_on_this_node) < args.das_peers_per_subprocess:
                client = client_queue.pop(0)
                clients_on_this_node.append(client)

            # Prepare the files and spawn the processes!
            out_file_path = os.path.join(os.getcwd(), "out_%d.log" % job_ind)
            peer_ids_str = ",".join(map(str, clients_on_this_node))

            train_cmd = [sys.executable] + sys.argv
            train_cmd += ["--das-peers-in-this-subprocess", peer_ids_str]
            bash_file_name = "run_%d.sh" % job_ind
            with open(bash_file_name, "w") as bash_file:
                bash_file.write("""#!/bin/bash
module load cuda11.7/toolkit/11.7
source /home/spandey/venv3/bin/activate
cd %s
export PYTHONPATH=%s
%s""" % (os.getcwd(), os.getcwd(), " ".join(train_cmd)))
                st = os.stat(bash_file_name)
                os.chmod(bash_file_name, st.st_mode | stat.S_IEXEC)

            cmd = "ssh fs3.das6.tudelft.nl \"prun -t 2:00:00 -np 1 -o %s %s\"" % (
                out_file_path, os.path.join(os.getcwd(), bash_file_name))
            logger.debug("Command: %s", cmd)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((p, cmd, clients_on_this_node))

        for p, cmd, clients in processes:
            p.wait()
            logger.info("Command %s completed (clients: %s)!", cmd, clients)
            if p.returncode != 0:
                raise RuntimeError("Training subprocess exited with non-zero code %d" % p.returncode)

            with open(os.path.join(data_path, "accuracies.csv"), "a") as all_accs_file:
                for client_id in clients:
                    with open(os.path.join(data_path, "accuracies_%d.csv" % client_id)) as acc_file:
                        content = acc_file.read()
                        all_accs_file.write(content)
