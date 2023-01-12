"""
Test to distill knowledge from a CIFAR10 pre-trained model to another one.
"""
import logging
import os

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

NUM_ROUNDS = 100 if "NUM_ROUNDS" not in os.environ else int(os.environ["NUM_ROUNDS"])
NUM_PEERS = 10

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cifar10_distillation")


class DatasetWithIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    learning_settings = LearningSettings(
        learning_rate=0.001 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
        momentum=0.9,
        batch_size=200,
        kd_temperature=6 if "TEMPERATURE" not in os.environ else int(os.environ["TEMPERATURE"]),
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

    with open(os.path.join("data", "accuracies.csv"), "w") as out_file:
        out_file.write("round,temperature,accuracy,loss\n")

    # Load the teacher models
    teacher_models = []
    teacher_models_dir = "data" if "TEACHER_MODELS_PATH" not in os.environ else os.environ["TEACHER_MODELS_PATH"]
    for n in range(NUM_PEERS):
        teacher_model = create_model("cifar10")
        teacher_model_path = os.path.join(teacher_models_dir, "cifar10_%d.model" % n)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=torch.device('cpu')))
        teacher_models.append(teacher_model)

    # Test accuracy
    data_dir = os.path.join(os.environ["HOME"], "dfl-data")

    mapping = Linear(1, 1)
    cifar10_testset = CIFAR10(0, 0, mapping, train_dir=data_dir, test_dir=data_dir)

    # Create the student model
    student_model = create_model("cifar10")
    trainer = ModelTrainer(data_dir, settings, 0)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    logger.debug("Device to train on: %s", device)
    for n in range(NUM_PEERS):
        teacher_models[n].to(device)
    student_model.to(device)

    # for n in range(NUM_PEERS):
    #     acc, loss = cifar10_testset.test(teacher_models[n], device_name=device)
    #     print("Teacher model %d accuracy: %f, loss: %f" % (n, acc, loss))

    train_set = trainer.dataset.get_trainset(batch_size=settings.learning.batch_size, shuffle=False)
    #train_set = get_random_images_data_loader(50000, settings.learning.batch_size)

    # Determine outputs of the teacher model on the public training data
    outputs = []
    for n in range(NUM_PEERS):
        teacher_outputs = []
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // settings.learning.batch_size
        if len(train_set.dataset) % settings.learning.batch_size != 0:
            local_steps += 1
        for local_step in range(local_steps):
            data, _ = next(train_set_it)
            data = Variable(data.to(device))
            out = teacher_models[n].forward(data)
            teacher_outputs += out

        outputs.append(teacher_outputs)
        print("Inferred %d outputs for teacher model %d" % (len(teacher_outputs), n))

    # Aggregate the predicted outputs
    print("Aggregating predictions...")
    aggregated_predictions = []
    for sample_ind in range(len(outputs[0])):
        predictions = [outputs[n][sample_ind] for n in range(NUM_PEERS)]
        aggregated_predictions.append(torch.mean(torch.stack(predictions), dim=0))

    train_set = DataLoader(DatasetWithIndex(train_set.dataset), batch_size=settings.learning.batch_size, shuffle=True)

    # Train the student model on the distilled outputs
    for epoch in range(NUM_ROUNDS):
        optimizer = SGDOptimizer(student_model, settings.learning.learning_rate, settings.learning.momentum)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // settings.learning.batch_size
        if len(train_set.dataset) % settings.learning.batch_size != 0:
            local_steps += 1
        logger.info("Will perform %d local steps", local_steps)
        for local_step in range(local_steps):
            data, _, indices = next(train_set_it)
            data = Variable(data.to(device))

            logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)

            teacher_output = torch.stack([aggregated_predictions[ind].detach().clone() for ind in indices])
            student_output = student_model.forward(data)
            loss = loss_fn_kd(student_output, teacher_output, learning_settings)

            optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.optimizer.step()

        acc, loss = cifar10_testset.test(student_model, device_name=device)
        print("Accuracy after %d epochs: %f, %f" % (epoch + 1, acc, loss))
        with open(os.path.join("data", "accuracies.csv"), "a") as out_file:
            out_file.write("%d,%d,%f,%f\n" % (epoch + 1, learning_settings.kd_temperature, acc, loss))
