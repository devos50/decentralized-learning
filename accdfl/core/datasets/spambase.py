import os
from random import Random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler

from accdfl.core.datasets.Dataset import Dataset
from accdfl.core.datasets.Partitioner import DirichletDataPartitioner, DataPartitioner, KShardDataPartitioner
from accdfl.core.mappings.Mapping import Mapping

NUM_CLASSES = 2


class Spambase(Dataset):
    """
    Class for the CIFAR10 dataset
    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        partitioner: str,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1024,
        shards=1,
        alpha: float = 1
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )

        self.partitioner = partitioner
        self.shards = shards
        self.alpha = alpha

        data = pd.read_csv(os.path.join(self.train_dir or self.test_dir, "spambase.data"), header=None).to_numpy()
        y = LabelEncoder().fit_transform(data[:, 57])
        x = np.delete(data, [57], axis=1).astype('float64')

        x = StandardScaler().fit_transform(x)
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()

        self.fullset = [(x[i], y[i]) for i in range(len(x))]
        Random(42).shuffle(self.fullset)

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

        # TODO: Add Validation

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        self.logger.info("Loading training set from directory %s and with alpha %f", self.train_dir, self.alpha)

        trainset = self.fullset[:4140]
        c_len = len(trainset)

        if self.sizes == None:  # Equal distribution of data among processes
            e = c_len // self.n_procs
            frac = e / c_len
            self.sizes = [frac] * self.n_procs
            self.sizes[-1] += 1.0 - frac * self.n_procs
            self.logger.debug("Size fractions: {}".format(self.sizes))

        self.uid = self.mapping.get_uid(self.rank, self.machine_id)

        if self.partitioner == "iid":
            self.trainset = DataPartitioner(trainset, self.sizes).use(self.uid)
        elif self.partitioner == "shards":
            train_data = {key: [] for key in range(10)}
            for x, y in trainset:
                train_data[y].append(x)
            all_trainset = []
            for y, x in train_data.items():
                all_trainset.extend([(a, y) for a in x])
            self.trainset = KShardDataPartitioner(all_trainset, self.sizes, shards=self.shards).use(self.uid)
        elif self.partitioner == "dirichlet":
            self.trainset = DirichletDataPartitioner(trainset, self.sizes, alpha=self.alpha).use(self.uid)
        else:
            raise RuntimeError("Unknown partitioner %s for CIFAR10 dataset", self.partitioner)

        self.logger.info("Train dataset initialization done! UID: %d. Total samples: %d", self.uid, len(self.trainset))

    def load_testset(self):
        """
        Loads the testing set.
        """
        self.logger.info("Loading testing set from data directory %s", self.test_dir)
        self.testset = self.fullset[4140:]
        self.logger.info("Test dataset initialization done! Total samples: %d", len(self.testset))

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            return DataLoader(self.testset, batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def test(self, model, device_name: str = "cpu"):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple
            (accuracy, loss_value)

        """

        testloader = self.get_testset()

        self.logger.debug("Test Loader instantiated.")
        device = torch.device(device_name)
        self.logger.debug("Device for Spambase accuracy check: %s", device)

        correct = example_number = total_loss = num_batches = 0
        model.to(device)
        model.eval()

        ce_loss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in iter(testloader):
                data, target = data.to(device), target.to(device)
                output = model.forward(data)
                if model.__class__.__name__ == "ResNet":
                    total_loss += ce_loss(output, target)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1

        accuracy = float(correct) / float(example_number) * 100.0
        loss = total_loss / float(example_number)
        return accuracy, loss
