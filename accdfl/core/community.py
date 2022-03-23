import hashlib
import io
import itertools
import json
import os
import random
from asyncio import Future, sleep
from binascii import unhexlify, hexlify
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from accdfl.core.blocks import ModelUpdateBlock
from accdfl.core.caches import DataRequestCache
from accdfl.core.model import serialize_model, unserialize_model
from accdfl.core.stores import DataStore, ModelStore, DataType
from accdfl.core.dataset import Dataset
from accdfl.core.listeners import ModelUpdateBlockListener
from accdfl.core.model.linear import LinearModel
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.payloads import DataRequest, DataNotFoundResponse
from accdfl.trustchain.community import TrustChainCommunity
from accdfl.util.eva_protocol import EVAProtocolMixin
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload, GlobalTimeDistributionPayload
from ipv8.util import fail


class DFLCommunity(EVAProtocolMixin, TrustChainCommunity):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eva_init()
        self.data_store = DataStore()
        self.model_store = ModelStore()
        self.compute_accuracy_after_averaging = False
        self.compute_accuracy_after_epoch = False
        self.model_performances = []
        self.total_samples_per_class = 5000
        self.model_send_delay = None
        self.round_complete_callback = None
        self.parameters = None
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.round = 1
        self.epoch = 1
        self.participants = None
        self.round_deferred = Future()
        self.incoming_models = {}

        self.eva_register_receive_callback(self.on_receive)
        self.eva_register_send_complete_callback(self.on_send_complete)
        self.eva_register_error_callback(self.on_error)

        self.model_update_block_listener = ModelUpdateBlockListener()
        self.add_listener(self.model_update_block_listener, [b"model_update"])

        self.add_message_handler(DataRequest, self.on_data_request)
        self.add_message_handler(DataNotFoundResponse, self.on_data_not_found_response)

        self.logger.info("The ADFL community started with public key: %s",
                         hexlify(self.my_peer.public_key.key_to_bin()).decode())

    def get_participant_index(self):
        if not self.participants:
            return -1
        return self.participants.index(hexlify(self.my_peer.public_key.key_to_bin()).decode())

    def setup(self, parameters):
        self.parameters = parameters
        self.model = LinearModel(28 * 28)  # For MNIST
        self.participants = parameters["participants"]
        self.logger.info("Setting up experiment with %d participants (I am participant %d)" %
                         (len(self.participants), self.get_participant_index()))

        self.dataset = Dataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters["batch_size"],
                               self.total_samples_per_class, len(self.participants), self.get_participant_index())
        self.optimizer = SGDOptimizer(self.model, parameters["learning_rate"], parameters["momentum"])

    async def train(self) -> bool:
        """
        Train the model on a batch. Return a boolean that indicates whether the epoch is completed.
        """
        old_model_serialized = serialize_model(self.model)
        old_model_hash = hexlify(hashlib.md5(old_model_serialized).digest()).decode()
        self.model_store.add(old_model_serialized)

        def it_has_next(iterable):
            try:
                first = next(iterable)
            except StopIteration:
                return None
            return itertools.chain([first], iterable)

        hashes = []
        data, target = self.dataset.iterator.__next__()
        for ddata, dtarget in zip(data, target):
            h = hashlib.md5(b"%d" % hash(ddata))
            hashes.append(hexlify(h.digest()).decode())
            self.data_store.add(ddata, dtarget)
        data, target = Variable(data), Variable(target)
        self.optimizer.optimizer.zero_grad()
        self.logger.info('d-sgd.next node forward propagation')
        output = self.model.forward(data)
        loss = F.nll_loss(output, target)
        self.logger.info('d-sgd.next node backward propagation')
        loss.backward()
        self.optimizer.optimizer.step()

        new_model_serialized = serialize_model(self.model)
        new_model_hash = hexlify(hashlib.md5(new_model_serialized).digest()).decode()
        self.model_store.add(new_model_serialized)

        # Record the training and aggregation step
        tx = {"round": self.round, "inputs": hashes, "old_model": old_model_hash, "new_model": new_model_hash}
        await self.self_sign_block(b"model_update", transaction=tx)

        # Are we at the end of the epoch?
        res = it_has_next(self.dataset.iterator)
        if res is None:
            self.epoch += 1
            self.logger.info("Epoch done - resetting dataset iterator")
            self.dataset.reset_iterator()
            return True
        else:
            self.dataset.iterator = res
            return False

    def average_models(self, models):
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            center_model = models[0].copy()
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model

    async def advance_round(self):
        """
        Complete a round of training and model aggregation.
        """
        # Train
        epoch_done = await self.train()

        # Send the model to your neighbours
        for peer in self.get_peers():
            response = {"round": self.round}
            if self.model_send_delay is not None:
                await sleep(random.randint(0, self.model_send_delay) / 1000)
            self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(self.model))

        if len(self.participants) > 1:
            await self.round_deferred

            self.logger.info("Received %d models from other peers - starting to average", len(self.incoming_models))

            # Average your model with those of the other participants
            avg_model = self.average_models(list(self.incoming_models.values()) + [self.model])
            with torch.no_grad():
                for p, new_p in zip(self.model.parameters(), avg_model.parameters()):
                    p.mul_(0.)
                    p.add_(new_p)

        if self.compute_accuracy_after_averaging or (epoch_done and self.compute_accuracy_after_epoch):
            self.logger.info("Computing accuracy of model for round %d (epoch: %d)", self.round, self.epoch)
            accuracy, loss = self.compute_accuracy()
            self.model_performances.append((self.round, accuracy, loss))

        self.logger.info("Round %d done", self.round)
        if self.round_complete_callback:
            self.round_complete_callback(self.round)
        self.round += 1
        self.round_deferred = Future()
        self.incoming_models = {}

    async def rounds(self):
        while self.round != self.parameters["rounds"] + 1:
            await self.advance_round()

    def compute_accuracy(self, max_items=-1):
        """
        Compute the accuracy/loss of the current model.
        """
        self.model.eval()
        correct = example_number = total_loss = num_batches = 0
        train = torch.utils.data.DataLoader(self.dataset.dataset, 100)
        with torch.no_grad():
            cur_item = 0
            for data, target in train:
                if max_items != -1 and cur_item == max_items:
                    break
                data, target = Variable(data), Variable(target)
                output = self.model.forward(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                num_batches += 1.0
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                example_number += target.size(0)
                cur_item += 1

        accuracy = float(correct) / float(example_number)
        loss = total_loss / float(example_number)
        return accuracy, loss

    def get_peer_by_pk(self, target_pk: bytes):
        peers = list(self.get_peers())
        for peer in peers:
            if peer.public_key.key_to_bin() == target_pk:
                return peer
        return None

    async def request_data(self, other_peer, data_hash: bytes, type=DataType.MODEL) -> Optional[bytes]:
        """
        Request data from another peer, based on a hash.
        """
        request_future = Future()
        cache = DataRequestCache(self, request_future)
        self.request_cache.add(cache)

        global_time = self.claim_global_time()
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = DataRequest(cache.number, data_hash, type.value)
        dist = GlobalTimeDistributionPayload(global_time)
        packet = self._ez_pack(self._prefix, DataRequest.msg_id, [auth, dist, payload])
        self.endpoint.send(other_peer.address, packet)

        return await request_future

    @lazy_wrapper(GlobalTimeDistributionPayload, DataRequest)
    def on_data_request(self, peer, dist, payload):
        request_type = DataType(payload.request_type)
        if request_type == DataType.TRAIN_DATA:
            request_data = self.data_store.get(payload.data_hash)
            if request_data:
                # Send the requested data to the requesting peer
                self.logger.debug("Sending data item with hash %s to peer %s", hexlify(payload.data_hash).decode(), peer)
                data, target = request_data
                b = io.BytesIO()
                torch.save(data, b)
                b.seek(0)
                response_data = json.dumps({
                    "hash": hexlify(payload.data_hash).decode(),
                    "request_id": payload.request_id,
                    "type": payload.request_type,
                    "target": int(target)
                }).encode()
                self.eva_send_binary(peer, response_data, b.read())
            else:
                self.send_data_not_found_message(peer, payload.data_hash, payload.request_id)
        elif request_type == DataType.MODEL:
            request_data = self.model_store.get(payload.data_hash)
            if request_data:
                self.logger.debug("Sending model with hash %s to peer %s", hexlify(payload.data_hash).decode(), peer)
                response_data = json.dumps({
                    "hash": hexlify(payload.data_hash).decode(),
                    "request_id": payload.request_id,
                    "type": payload.request_type
                }).encode()
                self.eva_send_binary(peer, response_data, request_data)
            else:
                self.send_data_not_found_message(peer, payload.data_hash, payload.request_id)

    def send_data_not_found_message(self, peer, data_hash, request_id):
        self.logger.warning("Data item %s requested by peer %s not found", hexlify(data_hash).decode(), peer)
        global_time = self.claim_global_time()
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())
        payload = DataNotFoundResponse(request_id)
        dist = GlobalTimeDistributionPayload(global_time)
        packet = self._ez_pack(self._prefix, DataNotFoundResponse.msg_id, [auth, dist, payload])
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper(GlobalTimeDistributionPayload, DataNotFoundResponse)
    def on_data_not_found_response(self, peer, _, payload):
        if not self.request_cache.has("datarequest", payload.request_id):
            self.logger.warning("Data request cache with ID %d not found!", payload.request_id)

        cache = self.request_cache.get("datarequest", payload.request_id)
        cache.received_not_found_response()

    def get_tc_record(self, peer_pk, round):
        """
        Look in the database for the record containing information associated with a model update in a particular round.
        """
        blocks = self.persistence.get_latest_blocks(peer_pk, limit=-1, block_types=[b"model_update"])
        for block in blocks:
            if block.public_key == peer_pk and block.transaction["round"] == round:
                return block
        return None

    def verify_model_training(self, old_model, data, target, new_model) -> bool:
        optimizer = SGDOptimizer(old_model, self.optimizer.learning_rate, self.optimizer.momentum)
        data, target = Variable(data), Variable(target)
        optimizer.optimizer.zero_grad()
        self.logger.info('d-sgd.next node forward propagation')
        output = old_model.forward(data)
        loss = F.nll_loss(output, target)
        self.logger.info('d-sgd.next node backward propagation')
        loss.backward()
        optimizer.optimizer.step()
        return torch.allclose(old_model.state_dict()["fc.weight"], new_model.state_dict()["fc.weight"])

    async def audit(self, other_peer_pk, round):
        """
        Audit the actions of another peer in a particular round.
        """
        # Get the TrustChain record associated with the other peer and a particular round
        block: ModelUpdateBlock = self.get_tc_record(other_peer_pk, round)
        if not block:
            return fail(RuntimeError("Could not find block associated with round %d" % round))

        # Request all inputs for a particular round
        other_peer = self.get_peer_by_pk(other_peer_pk)
        if not other_peer:
            return fail(RuntimeError("Could not find peer with public key %s" % hexlify(other_peer_pk)))

        datas = []
        targets = []
        for input_hash in block.inputs:
            data = self.data_store.get(input_hash)
            if not self.data_store.get(input_hash):
                data = await self.request_data(other_peer, input_hash, type=DataType.TRAIN_DATA)
                if not data:
                    return False

            # Convert data elements to Tensors
            target, data = data
            b = io.BytesIO(data)
            data = torch.load(b)
            self.data_store.add(data, target)
            datas.append(data.tolist())
            targets.append(target)

        # Fetch the model
        old_model_serialized = await self.request_data(other_peer, block.old_model, type=DataType.MODEL)
        old_model = unserialize_model(old_model_serialized)

        # TODO optimize this so we only compare the hash (avoid pulling in the new model)
        new_model_serialized = await self.request_data(other_peer, block.new_model, type=DataType.MODEL)
        new_model = unserialize_model(new_model_serialized)

        return self.verify_model_training(old_model, Tensor(datas), torch.LongTensor(targets), new_model)

    def on_receive(self, peer, binary_info, binary_data, nonce):
        self.logger.info(f'Data has been received from peer {peer}: {binary_info}')
        json_data = json.loads(binary_info.decode())
        if "request_id" in json_data:
            # We received this data in response to an earlier request
            if not self.request_cache.has("datarequest", json_data["request_id"]):
                self.logger.warning("Data request cache with ID %d not found!", json_data["request_id"])

            cache = self.request_cache.get("datarequest", json_data["request_id"])
            request_type = DataType(json_data["type"])
            if request_type == DataType.TRAIN_DATA:
                cache.request_future.set_result((json_data["target"], binary_data))
            elif request_type == DataType.MODEL:
                cache.request_future.set_result(binary_data)
        else:
            # This response is the model of another participant
            if json_data["round"] != self.round:
                self.logger.warning("Received model from peer %s for round %d but we are at round %d - ignoring model",
                                    peer, json_data["round"], self.round)
            else:
                incoming_model = unserialize_model(binary_data)
                self.incoming_models[peer.public_key.key_to_bin()] = incoming_model
                if len(self.incoming_models) == len(self.participants) - 1:
                    self.round_deferred.set_result(None)

    def on_send_complete(self, peer, binary_info, binary_data, nonce):
        self.logger.info(f'Outgoing transfer to peer {peer} has completed: {binary_info}')

    def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')
