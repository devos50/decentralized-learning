import csv
import os

import librosa
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from accdfl.core.datasets.Dataset import Dataset
from accdfl.core.datasets.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, \
    AddBackgroundNoiseOnSTFT, ToMelSpectrogramFromSTFT, DeleteSTFT
from accdfl.core.datasets.transforms_wav import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToTensor, \
    ToMelSpectrogram, LoadAudio
from accdfl.core.mappings.Mapping import Mapping
from accdfl.util.divide_data_fedscale import DataPartitioner, select_dataset

NUM_CLASSES = 35
CLASSES = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no', 'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward',
           'learn', 'house', 'three', 'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual', 'nine', 'bird', 'right', 'follow', 'bed']


class GoogleSpeech(Dataset):
    """
    Class for the Google Speech dataset
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

        self.classMapping = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.data_dir = os.path.join(self.train_dir or self.test_dir, "google_speech")
        self.data_file = "train" if self.__training__ else "test"
        self.training_sets = None
        self.testing_sets = None

        bkg = '_background_noise_'
        data_aug_transform = transforms.Compose(
            [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
             TimeshiftAudioOnSTFT(), FixSTFTDimension()])
        bg_dataset = BackgroundNoiseDataset(os.path.join(self.data_dir, bkg), data_aug_transform)
        add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
        train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
            n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
        self.train_transforms = transforms.Compose([LoadAudio(), data_aug_transform, add_bg_noise, train_feature_transform])

        valid_feature_transform = transforms.Compose(
            [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
        self.test_transforms = transforms.Compose([LoadAudio(), FixAudioLength(), valid_feature_transform])

        # load data and targets
        self.data, self.targets = self.load_files()

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def load_meta_data(self, path):
        data_to_label = {}
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    data_to_label[row[1]] = self.classMapping[row[-2]]
                line_count += 1

        return data_to_label

    def load_files(self):
        rawData, rawTags = [], []
        # load meta file to get labels
        classMapping = self.load_meta_data(os.path.join(
            self.data_dir, 'client_data_mapping', self.data_file + '.csv'))

        for imgFile in list(classMapping.keys()):
            rawData.append(imgFile)
            rawTags.append(classMapping[imgFile])

        return rawData, rawTags

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data[index], int(self.targets[index])
        data = {'path': os.path.join(self.data_dir, self.data_file, path), 'target': target}

        if self.__training__:
            data = self.train_transforms(data)
        else:
            data = self.test_transforms(data)

        return data['input'], data['target']

    def __len__(self):
        return len(self.data)

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.
        """
        self.logger.info("Loading training set from directory %s and with alpha %f", self.train_dir, self.alpha)

        self.training_sets = DataPartitioner(data=self, numOfClass=len(self.classMapping))
        self.training_sets.partition_data_helper(num_clients=self.n_procs)

        self.logger.info("Train dataset initialization done! UID: %d. Total samples: %d", self.uid, len(self.data))

    def load_testset(self):
        """
        Loads the testing set.
        """
        self.logger.info("Loading testing set from data directory %s", self.test_dir)

        self.testing_sets = DataPartitioner(data=self, numOfClass=len(self.classMapping), isTest=True)
        self.testing_sets.partition_data_helper(num_clients=1)

        self.logger.info("Test dataset initialization done! Total samples: %d", len(self.data))

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
            return select_dataset(self.rank, self.training_sets, batch_size=batch_size)
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
            return select_dataset(0, self.testing_sets, batch_size=self.test_batch_size)
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
        self.logger.debug("Device for Google Speech accuracy check: %s", device)

        correct = example_number = total_loss = num_batches = 0
        model.to(device)
        model.eval()

        ce_loss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in iter(testloader):
                data, target = data.to(device), target.to(device)
                data = torch.unsqueeze(data, 1)
                output = model.forward(data)
                total_loss += ce_loss(output, target)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1

        accuracy = float(correct) / float(example_number) * 100.0
        loss = total_loss / float(example_number)
        return accuracy, loss

    def get_num_classes(self):
        return NUM_CLASSES


class BackgroundNoiseDataset():
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index],
                'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data