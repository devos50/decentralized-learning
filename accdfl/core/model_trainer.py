import logging
import os
import time

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from accdfl.core.datasets import create_dataset
from accdfl.core.optimizer.sgd import SGDOptimizer


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, parameters, participant_index):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parameters = parameters

        if parameters["dataset"] in ["cifar10", "mnist", "movielens"]:
            train_dir = data_dir
        else:
            train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset = create_dataset(parameters, participant_index=participant_index, train_dir=train_dir)

    def train(self, model) -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        optimizer = SGDOptimizer(model, self.parameters["learning_rate"], self.parameters["momentum"])
        train_set = self.dataset.get_trainset(batch_size=self.parameters["batch_size"], shuffle=True)
        train_set_it = iter(train_set)
        local_steps = len(train_set.dataset) // self.parameters["batch_size"]
        if len(train_set.dataset) % self.parameters["batch_size"] != 0:
            local_steps += 1

        with open("time_stats.txt", "a") as time_stats:
            time_stats.write("about_to_start_train,%f\n" % time.time())

        self.logger.info("Will perform %d local steps of training (batch size: %d)", local_steps, self.parameters["batch_size"])

        for local_step in range(local_steps):
            with open("train_step.txt", "a") as progress_file:
                progress_file.write("%f,%d,%d\n" % (time.time(), local_step, local_steps))
            try:
                data, target = next(train_set_it)
                model.train()
                data, target = Variable(data), Variable(target)
                optimizer.optimizer.zero_grad()
                self.logger.info('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
                output = model.forward(data)

                if self.parameters["dataset"] == "movielens":
                    lossf = MSELoss()
                elif self.parameters["dataset"] == "cifar10":
                    lossf = NLLLoss()
                else:
                    lossf = CrossEntropyLoss()

                loss = lossf(output, target)
                self.logger.info('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
                loss.backward()
                optimizer.optimizer.step()
            except StopIteration:
                pass

        return model
