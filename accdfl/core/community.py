import hashlib
from binascii import unhexlify, hexlify

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from accdfl.core.dataset import Dataset
from accdfl.core.model.linear import LinearModel
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.trustchain.community import TrustChainCommunity
from accdfl.util.eva_protocol import EVAProtocolMixin


class DFLCommunity(EVAProtocolMixin, TrustChainCommunity):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eva_init()
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.round = 1

        self.eva_register_receive_callback(self.on_receive)
        self.eva_register_send_complete_callback(self.on_send_complete)
        self.eva_register_error_callback(self.on_error)

    def setup(self, parameters):
        self.model = LinearModel(28 * 28)  # For MNIST
        self.dataset = Dataset("/Users/martijndevos/Documents/mnist", parameters["batch_size"])
        self.optimizer = SGDOptimizer(self.model, parameters["learning_rate"], parameters["momentum"])

    async def train(self):
        old_model_hash = hash(self.model.parameters())
        hashes = []
        try:
            data, target = self.dataset.iterator.__next__()
            for ddata in data:
                h = hashlib.md5(b"%d" % hash(ddata))
                hashes.append(hexlify(h.digest()).decode())
            data, target = Variable(data), Variable(target)
            self.optimizer.optimizer.zero_grad()
            self.logger.info('d-sgd.next node forward propagation')
            output = self.model.forward(data)
            loss = F.nll_loss(output, target)
            self.logger.info('d-sgd.next node backward propagation')
            loss.backward()
            epoch_done = False
        except StopIteration:
            epoch_done = True

        if not epoch_done:
            self.optimizer.optimizer.step()

        new_model_hash = hash(self.model.parameters())

        # Record the training and aggregation step
        tx = {"round": self.round, "inputs": hashes, "old_model": old_model_hash, "new_model": new_model_hash}
        print(tx)
        await self.self_sign_block(b"model_update", transaction=tx)

    def compute_accuracy(self):
        self.model.eval()
        correct = example_number = total_loss = num_batches = 0
        train = torch.utils.data.DataLoader(self.dataset.dataset, 1000)
        with torch.no_grad():
            for data, target in train:
                data, target = Variable(data), Variable(target)
                output = self.model.forward(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                num_batches += 1.0
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                example_number += target.size(0)

        print(correct)
        print(example_number)

    def audit(self, other_peer_pk, round):
        """
        Audit the actions of another peer in a particular round.
        :return:
        """
        pass

    def on_receive(self, peer, binary_info, binary_data, nonce):
        self.logger.info(f'Data has been received: {binary_info}')

    def on_send_complete(self, peer, binary_info, binary_data, nonce):
        self.logger.info(f'Transfer has been completed: {binary_info}')

    def on_error(self, peer, exception):
        self.logger.error(f'Error has been occurred: {exception}')
