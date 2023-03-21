"""
Script to distill from n student models into a teacher model.
"""
import argparse
import asyncio
import logging
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from accdfl.core.datasets import create_dataset
from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.mappings import Linear
from accdfl.core.models import create_model
from accdfl.core.session_settings import SessionSettings, LearningSettings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distiller")


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
    parser.add_argument('models_dir')
    parser.add_argument('private_dataset')
    parser.add_argument('public_dataset')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--student-model', type=str, default=None)
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--acc-check-interval', type=int, default=1)
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()


async def run(args):
    # Initialize settings
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    settings = SessionSettings(
        dataset="cifar100",
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    with open(os.path.join("data", "distill_accuracies.csv"), "w") as out_file:
        out_file.write("epoch,accuracy,loss,best_acc,train_time,total_time\n")

    start_time = time.time()
    time_for_testing = 0  # Keep track of the time we spend on testing - we want to exclude this

    # Load the private testset and public dataset
    mapping = Linear(1, 1)
    cifar10_testset = CIFAR10(0, 0, mapping, "iid", test_dir=args.data_dir)
    cifar100_dataset = create_dataset(settings, train_dir=args.data_dir)
    cifar100_loader = DataLoader(dataset=cifar100_dataset.trainset, batch_size=args.batch_size, shuffle=False)

    # Load the teacher models
    teacher_models = []
    for i in range(args.peers):
        model_name = "%s_%d.model" % (args.private_dataset, i)
        model_path = os.path.join(args.models_dir, model_name)
        if not os.path.exists(model_path):
            raise RuntimeError("Could not find student model %d at location %s!" % (i, model_path))

        model = create_model(args.private_dataset, architecture=args.teacher_model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        teacher_models.append(model)

        # # Test accuracy of the teacher model
        # acc, loss = cifar10_testset.test(model, device_name=device)
        # logger.info("Accuracy of teacher model %d: %f, %f", i, acc, loss)

    # Create the student model
    student_model = create_model(args.private_dataset, architecture=args.student_model)
    student_model.to(device)

    # Generate the logits
    logits = []
    for teacher_ind, teacher_model in enumerate(teacher_models):
        teacher_logits = []
        for i, (images, _) in enumerate(cifar100_loader):
            images = images.to(device)
            with torch.no_grad():
                out = teacher_model.forward(images).detach()
            teacher_logits += out

        logits.append(teacher_logits)
        print("Inferred %d outputs for teacher model %d" % (len(teacher_logits), teacher_ind))

    # Aggregate the logits
    print("Aggregating logits...")
    aggregated_predictions = []
    for sample_ind in range(len(logits[0])):
        predictions = [logits[n][sample_ind] for n in range(args.peers)]
        aggregated_predictions.append(torch.mean(torch.stack(predictions), dim=0))

    # Reset loader
    cifar100_loader = DataLoader(dataset=DatasetWithIndex(cifar100_dataset.trainset), batch_size=args.batch_size, shuffle=True)

    # Distill \o/
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss(reduce=True)
    best_acc = 0
    for epoch in range(args.epochs):
        for i, (images, _, indices) in enumerate(cifar100_loader):
            images = images.to(device)

            student_model.train()
            teacher_logits = torch.stack([aggregated_predictions[ind].clone() for ind in indices])
            student_logits = student_model.forward(images)
            loss = criterion(teacher_logits, student_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute the accuracy of the student model
        if epoch % args.acc_check_interval == 0:
            test_start_time = time.time()
            acc, loss = cifar10_testset.test(student_model, device_name=device)
            if acc > best_acc:
                torch.save(student_model.state_dict(), os.path.join(args.models_dir, "best_distilled.model"))
                best_acc = acc
            logger.info("Accuracy of student model after %d epochs: %f, %f (best: %f)", epoch + 1, acc, loss, best_acc)
            time_for_testing += (time.time() - test_start_time)
            with open(os.path.join("data", "distill_accuracies.csv"), "a") as out_file:
                out_file.write("%d,%f,%f,%f,%f,%f\n" % (epoch + 1, acc, loss, best_acc, time.time() - start_time - time_for_testing, time.time() - start_time))

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(get_args()))
