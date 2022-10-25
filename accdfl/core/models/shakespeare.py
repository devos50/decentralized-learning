import torch.nn.functional as F
from torch import nn

from accdfl.core.datasets.Shakespeare import VOCAB_LEN
from accdfl.core.models.Model import Model


HIDDEN_DIM = 256
EMBEDDING_DIM = 8
NUM_LAYERS = 2
SEQ_LENGTH = 80


class LSTM(Model):
    """
    Class for a RNN Model for Sent140

    """

    def __init__(self):
        """
        Constructor. Instantiates the RNN Model to predict the next word of a sequence of word.
        Based on the TensorFlow model found here: https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        """
        super().__init__()

        # input_length does not exist
        self.embedding = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, num_layers=NUM_LAYERS
        )
        # activation function is added in the forward pass
        # Note: the tensorflow implementation did not use any activation function in this step?
        # should I use one.
        self.l1 = nn.Linear(HIDDEN_DIM * SEQ_LENGTH, VOCAB_LEN)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        # logging.info("Initial Shape: {}".format(x.shape))
        x = self.embedding(x)
        # logging.info("Embedding Shape: {}".format(x.shape))
        x, _ = self.lstm(x)
        # logging.info("LSTM Shape: {}".format(x.shape))
        x = F.relu(x.reshape((-1, HIDDEN_DIM * SEQ_LENGTH)))
        # logging.info("View Shape: {}".format(x.shape))
        x = self.l1(x)
        # logging.info("Output Shape: {}".format(x.shape))
        return x
