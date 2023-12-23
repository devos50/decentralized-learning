import os

import torch

from torchvision import transforms

from accdfl.core.datasets.google_speech import SPEECH, BackgroundNoiseDataset
from accdfl.core.datasets.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, \
    AddBackgroundNoiseOnSTFT, ToMelSpectrogramFromSTFT, DeleteSTFT
from accdfl.core.datasets.transforms_wav import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, LoadAudio, \
    ToMelSpectrogram, ToTensor
from accdfl.core.models.resnet_speech import resnet34

data_dir = "/Users/martijndevos/dfl-data/google_speech"

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

model = resnet34(num_classes=35, in_channels=1)
data_points = [train_dataset[i][0].unsqueeze(0) for i in range(4)]
batch = torch.stack(data_points, dim=0)
out = model(batch)
print(out)