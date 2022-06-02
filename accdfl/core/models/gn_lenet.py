from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNLeNet(nn.Module):
    """
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    Layer parameters taken from: https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/1parts/gnlenet_train_val.prototxt.template
    (with Group Normalisation).
    Results for previous model described in http://proceedings.mlr.press/v119/hsieh20a.html
    """

    def __init__(self, input_channel=3, output=10, model_input=(24, 24)):
        super(GNLeNet, self).__init__()

        self.input_channel = input_channel
        self.output = output
        self.model_input = model_input
        self.classifier_input = classifier_input_calculator(*model_input)

        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input, output),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def copy(self):
        c = GNLeNet(self.input_channel, self.output, self.model_input)
        for c1, s1 in zip(c.parameters(), self.parameters()):
            c1.mul_(0)
            c1.add_(s1)
        return c


def classifier_input_calculator(y, z):
    """Given the input shape of GN_Lenet, returns the size of the output of the Sequential module.
    This function is helpful to compute the size of the input of the classifier.
    Args:
        - y : the size of the first dimension of a channel in an input data
        - z : the size of the second dimension of a channel in an input data
    Output:
        - The size of the output of the sequential module of the GN_Lenet

    Example:
        Given an image from MNIST, since the images have shape (1,24,24), the size of a single
        channel is (24,24). classifier_input_calculator(24,24) == 256 and 256 is indeed the
        required input for the classifier.
    """

    def down(x, y, z):
        """Computes the final shape of a 3D tensor of shape (x,y,z) after the Conv2d and
        Maxpool layers in the gn_lenet model.

        Args:
            x (Int): 1st dimension of the input tensor
            y (Int): 2nd dimension
            z (Int): 3rd dimension

        Returns:
            (Int, Int, Int): Shape of the output of the Conv2d + Maxpool Layer.
        """
        return x, floor((y - 3) / 2) + 1, floor((z - 3) / 2) + 1

    x, y, z = down(32, y, z)
    x, y, z = down(x, y, z)
    x, y, z = down(2 * x, y, z)

    return x * y * z
