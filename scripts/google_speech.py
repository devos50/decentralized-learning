import logging
import os

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from torchvision import transforms

from accdfl.core.datasets.google_speech import SPEECH, BackgroundNoiseDataset
from accdfl.core.datasets.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, \
    AddBackgroundNoiseOnSTFT, ToMelSpectrogramFromSTFT, DeleteSTFT
from accdfl.core.datasets.transforms_wav import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, LoadAudio, \
    ToMelSpectrogram, ToTensor
from accdfl.core.models.resnet_speech import resnet34
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.util.divide_data import DataPartitioner, select_dataset

data_dir = "/Users/martijndevos/dfl-data/google_speech"

logging.basicConfig(level=logging.INFO)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res

def model_test(model, test_data, device):
    test_loss = 0
    correct = 0
    top_5 = 0
    test_len = 0
    with torch.no_grad():
        for data, target in test_data:
            data = torch.unsqueeze(data, 1)
            data, target = Variable(data.to(device)), Variable(target.to(device))

            output = model(data)
            lossf = CrossEntropyLoss()
            loss = lossf(output, target)

            test_loss += loss.data.item()  # Variable.data
            acc = accuracy(output, target, topk=(1, 5))

            correct += acc[0].item()
            top_5 += acc[1].item()
            test_len += len(target)

    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)

    return acc, acc_5

bkg = '_background_noise_'
data_aug_transform = transforms.Compose(
    [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
     TimeshiftAudioOnSTFT(), FixSTFTDimension()])
bg_dataset = BackgroundNoiseDataset(os.path.join(data_dir, bkg), data_aug_transform)
add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
    n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
train_dataset = SPEECH(data_dir, dataset='train',
                       transform=transforms.Compose([LoadAudio(),
                                                     data_aug_transform,
                                                     add_bg_noise,
                                                     train_feature_transform]))

valid_feature_transform = transforms.Compose(
    [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
test_dataset = SPEECH(data_dir, dataset='test',
                      transform=transforms.Compose([LoadAudio(),
                                                    FixAudioLength(),
                                                    valid_feature_transform]))

logging.info("Data partitioner starts ...")
training_sets = DataPartitioner(data=train_dataset, numOfClass=len(train_dataset.classMapping))
training_sets.partition_data_helper(num_clients=1)

testing_sets = DataPartitioner(data=test_dataset, numOfClass=len(test_dataset.classMapping), isTest=True)
testing_sets.partition_data_helper(num_clients=1)

client_data = select_dataset(1, training_sets, batch_size=32)
test_data = select_dataset(1, testing_sets, batch_size=128)

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info("Using device: %s" % device_name)
device = torch.device(device_name)
model = resnet34(num_classes=35, in_channels=1)
model = model.to(device)
logging.info(model_test(model, test_data, device))
optimizer = SGDOptimizer(model, 0.05, 0.9, weight_decay=0)
steps_done = 0
for data, target in client_data:
    data = torch.unsqueeze(data, 1)
    data, target = Variable(data.to(device)), Variable(target.to(device))
    lossf = CrossEntropyLoss()
    output = model.forward(data)
    loss = lossf(output, target)
    optimizer.optimizer.zero_grad()
    loss.backward()
    optimizer.optimizer.step()
    steps_done += 1
    logging.info("step %d done" % steps_done)

    if steps_done == 100:
        logging.info("Will test")
        test_results = model_test(model, test_data, device)
        logging.info(test_results)
