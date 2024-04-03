"""
Script to distill from n student models into a teacher model.
"""
import argparse
import asyncio
import glob
import logging
import os
import time
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from accdfl.core.datasets import create_dataset
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings
from accdfl.util.swa import SWA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distiller")

device = None
learning_settings = None
teacher_models = []
cohorts: Dict[int, List[int]] = {}
total_peers: int = 0
private_testset = None
weights = None
distill_timestamp = 0


class DatasetWithIndex(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_models_dir')  # The root directory containing the directories with data of individual (cohort) sessions
    parser.add_argument('models_base_name')  # The base name of the directories with data
    parser.add_argument('--cohort-file', type=str, default="cohorts.txt")
    parser.add_argument('private_dataset')
    parser.add_argument('public_dataset')
    parser.add_argument('--distill-timestamp', type=int, default=None)  # The timestamp during the experiment at which we distill
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "shards", "dirichlet", "realworld"])
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--student-model', type=str, default=None)
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--weighting-scheme', type=str, default="uniform", choices=["uniform", "label", "class"])
    parser.add_argument('--model-selection', type=str, default="best", choices=["best", "last"])
    parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--check-teachers-accuracy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--swa', action=argparse.BooleanOptionalAction)
    parser.add_argument('--private-data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    parser.add_argument('--public-data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cohort-participation', type=float, default=10)
    return parser.parse_args()


def read_cohorts(args) -> None:
    global total_peers
    logger.info("Reading cohort information...")

    # Read the cohort file
    with open(os.path.join(args.root_models_dir, args.cohort_file)) as cohort_file:
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
    data_dir = os.path.join(args.root_models_dir, args.models_base_name, "models")
    if not os.path.exists(data_dir):
        raise RuntimeError("Models directory %s does not exist!" % data_dir)

    for cohort_ind in range(len(cohorts.keys())):
        cohort_models = []

        # Gather all models in this directory
        for full_model_path in glob.glob("%s/c%d_*_%s.model" % (data_dir, cohort_ind, args.model_selection)):
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
        model = create_model(args.private_dataset, architecture=args.teacher_model)
        model.load_state_dict(torch.load(model_to_load, map_location=torch.device('cpu')))
        model.to(device)
        teacher_models.append(model)

        if args.check_teachers_accuracy:
            # Test accuracy of the teacher model
            acc, loss = private_testset.test(model, device_name=device)
            logger.info("Accuracy of teacher model %d: %f, %f", cohort_ind, acc, loss)

    distill_timestamp = args.distill_timestamp if args.distill_timestamp is not None else max(model_timestamps)


def determine_label_cohort_weights(args):
    global weights

    logger.info("Determining cohort weights...")

    # Determine the class distribution per cohort
    full_settings = SessionSettings(
        dataset=args.private_dataset,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=total_peers,
        partitioner=args.partitioner,
        alpha=args.alpha,
        seed=args.seed,
    )

    if full_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
        train_dir = args.private_data_dir
    else:
        train_dir = os.path.join(args.private_data_dir, "per_user_data", "train")

    # Get the number of classes in the dataset
    dataset = create_dataset(full_settings, 0, train_dir=train_dir)

    grouped_samples_per_class = []
    weights = []
    total_per_class = [0] * dataset.get_num_classes()
    for cohort_ind in range(len(cohorts.keys())):
        samples_per_class = [0] * dataset.get_num_classes()
        for peer_id in cohorts[cohort_ind]:
            start_time = time.time()

            dataset = create_dataset(full_settings, peer_id, train_dir=train_dir)
            logger.info("Creating dataset for peer %d took %f sec.", peer_id, time.time() - start_time)
            for a, (b, clsses) in enumerate(dataset.get_trainset(500, shuffle=False)):
                for cls in clsses:
                    samples_per_class[cls] += 1
                    total_per_class[cls] += 1
        logger.info("Samples per class for cohort %d: %s", cohort_ind, samples_per_class)
        grouped_samples_per_class.append(samples_per_class)

    logger.info("Total per class: %s", total_per_class)
    for cohort_ind in range(len(cohorts.keys())):
        weights_this_group = [grouped_samples_per_class[cohort_ind][i] / total_per_class[i] for i in range(dataset.get_num_classes())]
        weights.append(weights_this_group)


def determine_class_cohort_weights(args):
    global weights

    # Determine the class distribution per cohort
    full_settings = SessionSettings(
        dataset=args.private_dataset,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=total_peers,
        partitioner=args.partitioner,
        alpha=args.alpha,
        seed=args.seed,
    )

    if full_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
        train_dir = args.private_data_dir
    else:
        train_dir = os.path.join(args.private_data_dir, "per_user_data", "train")

    # Get the number of classes in the dataset
    dataset = create_dataset(full_settings, 0, train_dir=train_dir)

    assert len(cohorts) == dataset.get_num_classes(), "The number of cohorts should match the number of classes in the dataset"

    weights = []
    for cohort_ind in range(len(cohorts.keys())):
        weights_for_cohort = [0] * dataset.get_num_classes()
        weights_for_cohort[cohort_ind] = 1
        weights.append(weights_for_cohort)


def determine_cohort_weights(args):
    global weights

    if args.weighting_scheme == "uniform":
        weights = []
        num_cls = 10  # TODO hard-coded
        for cohort_ind in range(len(cohorts)):
            cohort_weights = [1 / num_cls] * num_cls
            weights.append(cohort_weights)
    elif args.weighting_scheme == "label":
        determine_label_cohort_weights(args)
    elif args.weighting_scheme == "class":
        determine_class_cohort_weights(args)
    else:
        raise RuntimeError("Unknown weight strategy %s" % args.weighting_scheme)

    for cohort_ind in range(len(weights)):
        logger.info("Weights for cohort %d: %s", cohort_ind, weights[cohort_ind])

    weights = torch.Tensor(weights)
    weights = weights.to(device)


async def run(args):
    global device, learning_settings, private_testset

    # Set the learning parameters if they are not set already
    if args.learning_rate is None:
        if args.public_dataset in ["cifar100", "stl10"]:
            args.learning_rate = 0.001
            args.momentum = 0.9
        elif args.public_dataset == "mnist" or args.public_dataset == "fashionmnist" or args.public_dataset == "svhn":
            args.learning_rate = 0.001
            args.momentum = 0
        else:
            raise RuntimeError("Unknown public dataset - unable set learning rate")

    read_cohorts(args)

    logger.info("Cohorts: %d, peers: %d" % (len(cohorts.keys()), total_peers))

    # Initialize settings
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_steps=0,  # Not used in our training
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    start_time = time.time()
    time_for_testing = 0  # Keep track of the time we spend on testing - we want to exclude this

    # Load the private testset and public dataset
    settings = SessionSettings(
        dataset=args.public_dataset,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    private_settings = SessionSettings(
        dataset=args.private_dataset,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    if private_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
        test_dir = args.private_data_dir
    else:
        test_dir = os.path.join(args.private_data_dir, "data", "test")

    if settings.dataset in ["cifar10", "cifar100", "stl10", "mnist", "fashionmnist", "svhn"]:
        train_dir = args.public_data_dir
    else:
        train_dir = os.path.join(args.public_data_dir, "per_user_data", "train")

    private_testset = create_dataset(private_settings, test_dir=test_dir)
    public_dataset = create_dataset(settings, train_dir=train_dir)
    public_dataset_loader = DataLoader(dataset=public_dataset.trainset, batch_size=args.batch_size, shuffle=False)

    read_teacher_models(args)

    if len(teacher_models) == 1:
        logger.info("Only one teacher model - no need for distillation")
        exit(0)

    determine_cohort_weights(args)

    # Create the student model
    student_model = create_model(args.private_dataset, architecture=args.student_model)
    student_model.to(device)

    # Generate the prediction logits that we will use to train the student model
    logits = []
    for teacher_ind, teacher_model in enumerate(teacher_models):
        logger.info("Inferring outputs for cohort %d model", teacher_ind)
        teacher_logits = []
        for i, (images, _) in enumerate(public_dataset_loader):
            images = images.to(device)
            with torch.no_grad():
                out = teacher_model.forward(images).detach()
                out *= weights[teacher_ind]
            teacher_logits += out

        logits.append(teacher_logits)
        logger.info("Inferred %d outputs for teacher model %d", len(teacher_logits), teacher_ind)

    # Aggregate the logits
    logger.info("Aggregating logits...")
    aggregated_predictions = []
    for sample_ind in range(len(logits[0])):
        predictions = [logits[n][sample_ind] for n in range(len(cohorts.keys()))]
        aggregated_predictions.append(torch.sum(torch.stack(predictions), dim=0))

    # Reset loader
    public_dataset_loader = DataLoader(dataset=DatasetWithIndex(public_dataset.trainset), batch_size=args.batch_size, shuffle=True)

    distill_results_file_name = "distill_accuracies_%d_%d_%f_%s_%.1f_%d_%s_%s.csv" % (len(cohorts), args.seed, args.learning_rate, args.optimizer, float(args.alpha), args.cohort_participation, args.private_dataset, args.public_dataset)
    with open(os.path.join("data", distill_results_file_name), "w") as out_file:
        out_file.write("cohorts,seed,learning_rate,optimizer,distill_time,public_dataset,weighting_scheme,epoch,iteration,accuracy,loss,best_acc,train_time,total_time\n")

    if args.optimizer == "adam":
        base_optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        base_optimizer = torch.optim.SGD(student_model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=0.00001)

    if args.swa:
        optimizer = SWA(base_optimizer, swa_start=500, swa_freq=25, swa_lr=None)
    else:
        optimizer = base_optimizer
    criterion = torch.nn.L1Loss(reduce=True)
    best_acc = 0
    iteration = 0

    def compute_student_accuracy():
        nonlocal best_acc, time_for_testing

        # Compute the accuracy of the student model
        test_start_time = time.time()
        acc, loss = private_testset.test(student_model, device_name=device)
        if acc > best_acc:
            best_acc = acc
        logger.info("Accuracy of student model after %d epochs (%d iterations): %f, %f (best: %f)", epoch + 1, iteration, acc, loss, best_acc)
        time_for_testing += (time.time() - test_start_time)
        with open(os.path.join("data", distill_results_file_name), "a") as out_file:
            out_file.write("%d,%d,%f,%s,%d,%s,%s,%d,%d,%f,%f,%f,%f,%f\n" % (len(cohorts), args.seed, args.learning_rate,
            args.optimizer, distill_timestamp, args.public_dataset, args.weighting_scheme, epoch + 1, iteration, acc,
            loss, best_acc, time.time() - start_time - time_for_testing, time.time() - start_time))

    for epoch in range(args.epochs):
        for i, (images, _, indices) in enumerate(public_dataset_loader):
            images = images.to(device)

            student_model.train()
            teacher_logits = torch.stack([aggregated_predictions[ind].clone() for ind in indices])
            student_logits = student_model.forward(images)
            loss = criterion(teacher_logits, student_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1

        compute_student_accuracy()

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(get_args()))
