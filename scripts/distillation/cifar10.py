"""
Test to distill knowledge from a CIFAR10 pre-trained model to another one.
"""
import asyncio
import copy
import logging
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from accdfl.core.datasets.CIFAR10 import CIFAR10
from accdfl.core.mappings import Linear
from accdfl.core.model_trainer import ModelTrainer
from accdfl.core.models import create_model
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.session_settings import LearningSettings, SessionSettings

from scripts.distillation import loss_fn_kd

NUM_ROUNDS = 10 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])
NUM_LOCAL_TRAIN_ROUNDS = 10 if "NUM_LOCAL_TRAIN_ROUNDS" not in os.environ else int(os.environ["NUM_LOCAL_TRAIN_ROUNDS"])
NUM_DISTILLATION_ROUNDS = 10 if "NUM_DISTILLATION_ROUNDS" not in os.environ else int(os.environ["NUM_DISTILLATION_ROUNDS"])
NUM_PEERS = 10 if "NUM_PEERS" not in os.environ else int(os.environ["NUM_PEERS"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cifar10_distillation")


class DatasetWithIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


async def run():
    # Initialize the settings
    train_learning_settings = LearningSettings(
        learning_rate=0.001 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
        momentum=0.9,
        batch_size=200,
    )

    train_settings = SessionSettings(
        dataset="cifar10",
        work_dir="",
        learning=train_learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=10,
    )

    distill_learning_settings = LearningSettings(
        learning_rate=0.1 if "DISTILL_LEARNING_RATE" not in os.environ else float(os.environ["DISTILL_LEARNING_RATE"]),
        momentum=0.9,
        batch_size=200,
        kd_temperature=6 if "TEMPERATURE" not in os.environ else int(os.environ["TEMPERATURE"]),
    )

    distill_settings = SessionSettings(
        dataset="cifar100",
        work_dir="",
        learning=distill_learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=1,
    )

    if not os.path.exists("data"):
        os.mkdir("data")

    with open(os.path.join("data", "accuracies.csv"), "w") as out_file:
        out_file.write("round,temperature,accuracy,loss\n")

    # Load the teacher models
    teacher_models = []
    for n in range(NUM_PEERS):
        teacher_model = create_model("cifar10")
        teacher_models.append(teacher_model)

    mapping = Linear(1, 1)
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")
    cifar10_testset = CIFAR10(0, 0, mapping, test_dir=data_dir)
    cifar100_trainer = ModelTrainer(data_dir, distill_settings, 0)
    cifar100_train_set = cifar100_trainer.dataset.get_trainset(batch_size=distill_settings.learning.batch_size, shuffle=False)
    trainers = [ModelTrainer(data_dir, train_settings, n) for n in range(NUM_PEERS)]

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    logger.debug("Device to train on: %s", device)
    for n in range(NUM_PEERS):
        teacher_models[n].to(device)

    # for n in range(NUM_PEERS):
    #     acc, loss = cifar10_testset.test(teacher_models[n], device_name=device)
    #     print("Teacher model %d accuracy: %f, loss: %f" % (n, acc, loss))

    for round_nr in range(NUM_ROUNDS):
        # Step 1) train all client models using their private data for one epoch
        for n in range(NUM_PEERS):
            for r in range(NUM_LOCAL_TRAIN_ROUNDS):
                start_time = time.time()
                await trainers[n].train(teacher_models[n], device_name=device)
                logger.info("Training for peer %d and round %d done - time: %f", n, r, time.time() - start_time)

            acc, loss = cifar10_testset.test(teacher_models[n], device_name=device)
            logger.info("Accuracy of teacher model %d after %d rounds: %f, %f", n, round_nr + 1, acc, loss)

        logger.info("Training for all %d clients done!", NUM_PEERS)

        # Step 2) determine outputs of the teacher model on the public training data
        outputs = []
        logger.info("Starting to compute inferences for %d peers", NUM_PEERS)
        for n in range(NUM_PEERS):
            teacher_outputs = []
            train_set_it = iter(cifar100_train_set)
            local_steps = len(cifar100_train_set.dataset) // distill_settings.learning.batch_size
            if len(cifar100_train_set.dataset) % distill_settings.learning.batch_size != 0:
                local_steps += 1
            for local_step in range(local_steps):
                data, _ = next(train_set_it)
                data = Variable(data.to(device))
                out = teacher_models[n].forward(data).detach()
                teacher_outputs += out

            outputs.append(teacher_outputs)
            logger.info("Inferred %d outputs for teacher model %d", len(teacher_outputs), n)

        # Step 3) aggregate the predicted outputs
        logger.info("Aggregating predictions...")
        aggregated_predictions = []
        for sample_ind in range(len(outputs[0])):
            predictions = [outputs[n][sample_ind] for n in range(NUM_PEERS)]
            aggregated_predictions.append(torch.mean(torch.stack(predictions), dim=0))

        train_set = DataLoader(DatasetWithIndex(cifar100_train_set.dataset), batch_size=distill_settings.learning.batch_size, shuffle=True)

        student_model = create_model("cifar10")
        student_model.to(device)

        # Step 4) train a student model on the distilled outputs for some rounds
        for epoch in range(NUM_DISTILLATION_ROUNDS):
            optimizer = SGDOptimizer(student_model, distill_settings.learning.learning_rate, distill_settings.learning.momentum)
            train_set_it = iter(train_set)
            local_steps = len(train_set.dataset) // distill_settings.learning.batch_size
            if len(train_set.dataset) % distill_settings.learning.batch_size != 0:
                local_steps += 1
            logger.info("Will perform %d local steps", local_steps)
            for local_step in range(local_steps):
                data, _, indices = next(train_set_it)
                data = Variable(data.to(device))

                logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)

                teacher_output = torch.stack([aggregated_predictions[ind].clone() for ind in indices])
                student_output = student_model.forward(data)
                loss = loss_fn_kd(student_output, teacher_output, distill_learning_settings)

                optimizer.optimizer.zero_grad()
                loss.backward()
                optimizer.optimizer.step()

        # Step 5) compute the accuracy of the global student model
        acc, loss = cifar10_testset.test(student_model, device_name=device)
        logger.info("Accuracy of global student model after %d rounds: %f, %f", round_nr + 1, acc, loss)
        with open(os.path.join("data", "accuracies.csv"), "a") as out_file:
            out_file.write("%d,%d,%f,%f\n" % (round_nr + 1, distill_learning_settings.kd_temperature, acc, loss))

        # Step 6) replace the local teacher models with the student model
        for n in range(NUM_PEERS):
            teacher_models[n] = copy.deepcopy(student_model)
            teacher_models[n].to(device)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
