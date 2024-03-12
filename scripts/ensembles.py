import argparse
import asyncio
import glob
import logging
import os
import shutil
import statistics
import time
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from accdfl.core.datasets import create_dataset
from accdfl.core.models import create_model
from accdfl.core.models.akashnet import AkashNet
from accdfl.core.session_settings import LearningSettings, SessionSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensembler")

data_dir = None
device = None
test_learning_settings = None
train_learning_settings = None
teacher_models = []
cohorts: Dict[int, List[int]] = {}
total_peers: int = 0
testset = None
akashnet = None
test_inputs = None
test_targets = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_models_dir')  # The root directory containing the directories with data of individual (cohort) sessions
    parser.add_argument('models_base_name')  # The base name of the directories with data
    parser.add_argument('--cohort-file', type=str, default="cohorts.txt")
    parser.add_argument('dataset')
    parser.add_argument('--distill-timestamp', type=int, default=None)  # The timestamp during the experiment at which we distill
    parser.add_argument('--partitioner', type=str, default="iid")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--train-batch-size', type=int, default=20)
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--check-teachers-accuracy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    parser.add_argument('--model-selection', type=str, default="best", choices=["best", "last"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-set-fraction', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--test-interval', type=int, default=50)
    return parser.parse_args()


def read_cohorts(args) -> None:
    global total_peers
    logger.info("Reading cohort information...")

    # Read the cohort file
    with open(os.path.join(args.root_models_dir, args.cohort_file), "r") as cohort_file:
        for cohort_line in cohort_file.readlines():
            parts = cohort_line.strip().split(",")
            cohort_index = int(parts[0])
            cohort_peers = [int(p) for p in parts[1].split("-")]
            cohorts[cohort_index] = cohort_peers
            total_peers += len(cohort_peers)


def read_teacher_models(args):
    global distill_timestamp
    logger.info("Reading teacher models...")

    model_timestamps: List[int] = []

    # Load the teacher models
    models_data_dir = os.path.join(args.root_models_dir, args.models_base_name, "models")
    if not os.path.exists(models_data_dir):
        raise RuntimeError("Models directory %s does not exist!" % models_data_dir)

    for cohort_ind in range(len(cohorts.keys())):
        cohort_models = []

        # Gather all models in this directory
        for full_model_path in glob.glob("%s/c%d_*_%s.model" % (models_data_dir, cohort_ind, args.model_selection)):
            model_name = os.path.basename(full_model_path).split(".")[0]
            parts = model_name.split("_")
            model_round = int(parts[1])
            model_time = int(parts[2])
            cohort_models.append((model_round, model_time, full_model_path))

        # Sort the models based on their timestamp
        cohort_models.sort(key=lambda x: x[1])

        # Find the right model given the distillation timestamp
        if args.distill_timestamp is None:
            model_to_load = cohort_models[-1][2]
            model_timestamps.append(cohort_models[-1][1])
        else:
            highest_ind = None
            for ind in range(len(cohort_models)):
                if highest_ind is None or cohort_models[ind][1] <= args.distill_timestamp:
                    highest_ind = ind

            model_to_load = cohort_models[highest_ind][2]
            model_timestamps.append(cohort_models[highest_ind][1])

        logger.info("Using model %s for cohort %d", os.path.basename(model_to_load), cohort_ind)
        model = create_model(args.dataset, architecture=args.teacher_model)
        model.load_state_dict(torch.load(model_to_load, map_location=torch.device('cpu')))
        model.to(device)
        teacher_models.append(model)

        if args.check_teachers_accuracy:
            # Test accuracy of the teacher model
            acc, loss = testset.test(model, device_name=device)
            logger.info("Accuracy of teacher model %d: %f, %f", cohort_ind, acc, loss)

    distill_timestamp = args.distill_timestamp if args.distill_timestamp is not None else max(model_timestamps)


def generate_logits(args):
    full_settings = SessionSettings(
        dataset=args.dataset,
        work_dir="",
        learning=test_learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=total_peers,
        partitioner=args.partitioner,
        alpha=args.alpha,
        seed=args.seed,
        validation_set_fraction=args.validation_set_fraction,
    )

    if full_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
        train_dir = args.data_dir
    else:
        train_dir = os.path.join(args.data_dir, "per_user_data", "train")

    # Load the datasets
    train_inputs = None
    train_targets = None
    for peer_id in range(total_peers):
        start_time = time.time()
        dataset = create_dataset(full_settings, peer_id, train_dir=train_dir)
        logger.info("Creating dataset for peer %d took %f sec.", peer_id, time.time() - start_time)

        validation_set = dataset.get_validationset()
        with torch.no_grad():
            has_samples = False
            for data, target in validation_set:
                has_samples = True
                if train_targets is None:
                    train_targets = target
                else:
                    train_targets = torch.cat((train_targets, target))

                data, target = data.to(device), target.to(device)
                batch_outputs = []
                for teacher_ind, teacher_model in enumerate(teacher_models):
                    output = teacher_model(data)
                    batch_outputs.append(output)

            if has_samples:
                cat_outputs = torch.cat(batch_outputs, 1)
                if train_inputs is None:
                    train_inputs = cat_outputs
                else:
                    train_inputs = torch.cat((train_inputs, cat_outputs))

    return train_inputs, train_targets


def generate_test_logits():
    global test_inputs, test_targets

    logger.info("Generating test logits...")
    testloader = testset.get_testset()

    with torch.no_grad():
        for data, target in iter(testloader):
            if test_targets is None:
                test_targets = target
            else:
                test_targets = torch.cat((test_targets, target))

            data, target = data.to(device), target.to(device)
            batch_outputs = []
            for teacher_ind, teacher_model in enumerate(teacher_models):
                output = teacher_model(data)
                batch_outputs.append(output)

            cat_outputs = torch.cat(batch_outputs, 1)
            if test_inputs is None:
                test_inputs = cat_outputs
            else:
                test_inputs = torch.cat((test_inputs, cat_outputs))


def evaluate_akashnet(args):
    if test_targets is None:
        generate_test_logits()
        print("Dimension of test inputs: %s" % str(test_inputs.size()))
        print("Dimension of test targets: %s" % str(test_targets.size()))

    dataset = TensorDataset(test_inputs, test_targets)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    correct = example_number = total_loss = num_batches = 0
    akashnet.to(device)
    akashnet.eval()

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in iter(data_loader):
            data, target = data.to(device), target.to(device)
            output = akashnet.forward(data)
            total_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            example_number += target.size(0)
            num_batches += 1

    accuracy = float(correct) / float(example_number) * 100.0
    loss = total_loss / float(example_number)
    return accuracy, loss


def train_akashnet(args, train_inputs, train_targets):
    global akashnet, data_dir
    logger.info("Starting to train AkashNet with %d samples", len(train_targets))

    dataset = TensorDataset(train_inputs, train_targets)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    optimizer = Adam(akashnet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    criterion = torch.nn.CrossEntropyLoss()
    akashnet.train()
    model = akashnet.to(device)
    train_loss_file_path = os.path.join(data_dir, "train_loss.csv")
    train_loss_file = open(train_loss_file_path, "w")
    train_loss_file.write("step,loss\n")

    step: int = 0
    for epoch in range(args.epochs):
        epoch_losses: List[float] = []

        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            # forward pass
            output = model(data)
            loss = criterion(output, target)
            epoch_losses.append(loss.item())

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()

            step += 1

        scheduler.step()

        mean_train_loss = statistics.mean(epoch_losses)
        train_loss_file.write("%d,%f\n" % (epoch + 1, mean_train_loss))
        logger.info("Training loss of epoch %d: %f" % (epoch + 1, mean_train_loss))

        if epoch % args.test_interval == 0:
            accuracy, loss = evaluate_akashnet(args)
            logger.info("Evaluation @ epoch %d: acc %f, loss %f" % (epoch, accuracy, loss))

    train_loss_file.close()


async def run(args):
    global device, test_learning_settings, train_learning_settings, testset, akashnet, data_dir

    read_cohorts(args)

    logger.info("Cohorts: %d, peers: %d" % (len(cohorts.keys()), total_peers))

    data_dir = os.path.join(args.root_models_dir, "ensembles_%s" % args.dataset)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Initialize settings
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    test_learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.test_batch_size,
        local_steps=0,
    )

    train_learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.train_batch_size,
        local_steps=0,
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    test_settings = SessionSettings(
        dataset=args.dataset,
        work_dir="",
        learning=test_learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    if test_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
        test_dir = args.data_dir
    else:
        test_dir = os.path.join(args.data_dir, "data", "test")

    testset = create_dataset(test_settings, test_dir=test_dir)

    read_teacher_models(args)

    if len(teacher_models) == 1:
        logger.info("Only one teacher model - no need for ensembles")
        exit(0)

    train_inputs, train_targets = generate_logits(args)
    print("Dimension of train inputs: %s" % str(train_inputs.size()))
    print("Dimension of train targets: %s" % str(train_targets.size()))

    num_classes = 10 if args.dataset == "cifar10" else 62
    akashnet = AkashNet(total_clients=len(cohorts), num_classes=num_classes)
    train_akashnet(args, train_inputs, train_targets)

loop = asyncio.get_event_loop()
loop.run_until_complete(run(get_args()))
