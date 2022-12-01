import torch

from accdfl.core.models.Model import Model


class MatrixFactorization(Model):
    """
    Class for a Matrix Factorization model for MovieLens.
    """

    def __init__(self, n_users=610, n_items=9724, n_factors=20):
        """
        Instantiates the Matrix Factorization model with user and item embeddings.

        Parameters
        ----------
        n_users
            The number of unique users.
        n_items
            The number of unique items.
        n_factors
            The number of columns in embeddings matrix.
        """
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors.weight.data.uniform_(-0.05, 0.05)

        self.detected_cuda: bool = False
        self.requires_cuda: bool = False

    def get_cuda_tensors(self, data):
        return torch.cuda.LongTensor(data[:, 0]) - 1, torch.cuda.LongTensor(data[:, 1]) - 1

    def get_cpu_tensors(self, data):
        return torch.LongTensor(data[:, 0]) - 1, torch.LongTensor(data[:, 1]) - 1

    def forward(self, data):
        """
        Forward pass of the model, it does matrix multiplication and returns predictions for given users and items.
        """
        if self.detected_cuda:
            users, items = self.get_cuda_tensors(data) if self.requires_cuda else self.get_cpu_tensors(data)
        else:
            try:
                # Try CPU tensors first. If it fails, attempt to instantiate CUDA tensors.
                users, items = self.get_cpu_tensors(data)
            except TypeError:
                users, items = self.get_cuda_tensors(data)
                self.requires_cuda = True
            self.detected_cuda = True

        u, it = self.user_factors(users), self.item_factors(items)
        x = (u * it).sum(dim=1, keepdim=True)
        return x.squeeze(1)
