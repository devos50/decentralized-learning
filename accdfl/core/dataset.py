import torch

from torchvision import datasets, transforms


class Dataset:

    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform)

        self.iterator = iter(torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        ))
