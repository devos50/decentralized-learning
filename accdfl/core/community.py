import hashlib
import io
import itertools
import json
import os
import random
from asyncio import Future, sleep, ensure_future
from binascii import unhexlify, hexlify
from enum import Enum
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from accdfl.core.blocks import ModelUpdateBlock
from accdfl.core.caches import DataRequestCache
from accdfl.core.model import serialize_model, unserialize_model, create_model, ModelType
from accdfl.core.stores import DataStore, ModelStore, DataType
from accdfl.core.dataset import Dataset
from accdfl.core.listeners import ModelUpdateBlockListener
from accdfl.core.optimizer.sgd import SGDOptimizer
from accdfl.core.payloads import DataRequest, DataNotFoundResponse, ModelTorrent
from accdfl.core.torrent_download_manager import TorrentDownloadManager
from accdfl.test.util.network_utils import NetworkUtils
from accdfl.trustchain.community import TrustChainCommunity
from accdfl.util.eva_protocol import EVAProtocolMixin, TransferResult
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_headers import BinMemberAuthenticationPayload, GlobalTimeDistributionPayload
from ipv8.util import fail


class TransmissionMethod(Enum):
    EVA = 0
    LIBTORRENT = 1


class DFLCommunity(EVAProtocolMixin, TrustChainCommunity):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_active = False
        self.is_participating_in_round = False
        self.data_store = DataStore()
        self.model_store = ModelStore()
        self.model_send_delay = None
        self.round_complete_callback = None
        self.parameters = None
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.round = 1
        self.epoch = 1
        self.sample_size = None
        self.participants: Optional[List[str]] = None
        self.round_deferred = Future()
        self.incoming_local_models: Dict[int, List] = {}
        self.incoming_aggregated_models: Dict[int, List] = {}
        self.is_computing_accuracy = False
        self.compute_accuracy_deferred = None

        # Model exchange parameters
        self.is_local_test = False  # To make sure that the libtorrent connect_peer works
        self.data_dir = None
        self.transmission_method = TransmissionMethod.EVA
        self.eva_max_retry_attempts = 20
        self.lt_listen_port = NetworkUtils().get_random_free_port()
        self.torrent_download_manager: Optional[TorrentDownloadManager] = None

        self.model_update_block_listener = ModelUpdateBlockListener()
        self.add_listener(self.model_update_block_listener, [b"model_update"])

        self.add_message_handler(DataRequest, self.on_data_request)
        self.add_message_handler(DataNotFoundResponse, self.on_data_not_found_response)
        self.add_message_handler(ModelTorrent, self.on_model_torrent)

        self.logger.info("The ADFL community started with public key: %s",
                         hexlify(self.my_peer.public_key.key_to_bin()).decode())

    def start(self):
        """
        Start to participate in the training process.
        """
        self.is_active = True

        # Start the process
        if self.is_participant_for_round(self.round):
            ensure_future(self.participate_in_round())
        else:
            self.logger.info("Participant %d won't participate in round %d", self.get_my_participant_index(), self.round)

    def get_participant_index(self, public_key: bytes) -> int:
        if not self.participants:
            return -1
        return self.participants.index(hexlify(public_key).decode())

    def get_my_participant_index(self):
        return self.get_participant_index(self.my_peer.public_key.key_to_bin())

    def get_participants_for_round(self, round: int) -> List[int]:
        rand = random.Random(round)
        participant_indices = list(range(len(self.participants)))
        return sorted(rand.sample(participant_indices, self.sample_size))

    def is_participant_for_round(self, round: int) -> bool:
        return self.get_my_participant_index() in self.get_participants_for_round(round)

    def get_round_representative(self, round: int) -> int:
        rand = random.Random(round)
        return rand.choice(self.get_participants_for_round(round))

    def is_round_representative(self, round: int) -> bool:
        return self.get_my_participant_index() == self.get_round_representative(round)

    def setup(self, parameters: Dict, data_dir: str, transmission_method: TransmissionMethod = TransmissionMethod.EVA):
        assert len(parameters["participants"]) * parameters["local_classes"] == sum(parameters["nodes_per_class"])

        self.parameters = parameters
        self.data_dir = data_dir
        self.sample_size = parameters["sample_size"]
        self.model = create_model(parameters["dataset"], parameters["model"])
        self.participants = parameters["participants"]
        self.logger.info("Setting up experiment with %d participants and sample size %d (I am participant %d)" %
                         (len(self.participants), self.sample_size, self.get_my_participant_index()))

        self.dataset = Dataset(os.path.join(os.environ["HOME"], "dfl-data"), parameters, self.get_my_participant_index())
        self.optimizer = SGDOptimizer(self.model, parameters["learning_rate"], parameters["momentum"])

        # Setup the model transmission
        self.transmission_method = transmission_method
        if self.transmission_method == TransmissionMethod.EVA:
            self.logger.info("Setting up EVA protocol")
            self.eva_init(window_size_in_blocks=32, retransmit_attempt_count=10, retransmit_interval_in_sec=1, timeout_interval_in_sec=10)
            self.eva_register_receive_callback(self.on_receive)
            self.eva_register_send_complete_callback(self.on_send_complete)
            self.eva_register_error_callback(self.on_error)
        else:
            self.logger.info("Setting up libtorrent transmission engine")
            self.torrent_download_manager = TorrentDownloadManager(self.data_dir, self.get_my_participant_index())
            self.torrent_download_manager.start(self.lt_listen_port)

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
        self.model.train()
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

        # Record the individual model update
        tx = {"round": self.round, "old_model": old_model_hash, "new_model": new_model_hash}
        #await self.self_sign_block(b"model_update", transaction=tx)

        # Are we at the end of the epoch?
        res = it_has_next(self.dataset.iterator)
        if res is None:
            self.epoch += 1
            self.logger.info("Epoch done - resetting dataset iterator")
            self.dataset.reset_train_iterator()
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

    async def send_aggregated_model(self, round, model):
        """
        Send the global model update to the participants of the next round.
        """
        participants_next_round = self.get_participants_for_round(round + 1)
        for participant_ind in participants_next_round:
            if participant_ind == self.get_my_participant_index():
                continue

            participant_pk = unhexlify(self.participants[participant_ind])
            peer = self.get_peer_by_pk(participant_pk)
            if not peer:
                self.logger.warning("Peer object of participant %d not available - not sending aggregated model", participant_ind)
                continue

            # TODO do something with the TrustChain block
            if self.transmission_method == TransmissionMethod.EVA:
                await self.eva_send_aggregated_model(round, model, peer)
            elif self.transmission_method == TransmissionMethod.LIBTORRENT:
                await sleep(random.random() / 4)  # Make sure we are not sending the torrents at exactly the same time
                await self.lt_send_aggregated_model(model, peer)

    async def eva_send_aggregated_model(self, round, model, peer):
        response = {"round": round, "type": "aggregated_model"}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            self.logger.info("Participant %d sending round %d aggregated model to peer %s (attempt %d)",
                             self.get_my_participant_index(), round, peer, attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(model))
                self.logger.info("Aggregated model of round %d successfully sent to peer %s", round, peer)
                break
            except Exception:
                self.logger.exception("Exception when sending aggregated model to peer %s", peer)
            attempt += 1

    async def lt_send_aggregated_model(self, model, peer):
        if not self.torrent_download_manager.is_seeding(self.round, ModelType.AGGREGATED):
            await self.torrent_download_manager.seed(self.round, ModelType.AGGREGATED, model)

        bencoded_torrent = self.torrent_download_manager.get_torrent_info(
            self.get_my_participant_index(), self.round, ModelType.AGGREGATED)
        self.send_model_torrent(peer, self.round, ModelType.AGGREGATED, bencoded_torrent)

    async def send_local_model(self):
        """
        Send the global model to the round representative.
        """
        if self.is_round_representative(self.round):
            return

        round_representative = self.get_round_representative(self.round)
        participant_pk = unhexlify(self.participants[round_representative])
        peer = self.get_peer_by_pk(participant_pk)
        if not peer:
            self.logger.warning("Peer object of round representative %d not available - not sending local model", round_representative)
            return

        if self.transmission_method == TransmissionMethod.EVA:
            await self.eva_send_local_model(peer)
        elif self.transmission_method == TransmissionMethod.LIBTORRENT:
            await sleep(random.random() / 4)  # Make sure we are not sending the torrents at exactly the same time
            await self.lt_send_local_model(peer)

    async def eva_send_local_model(self, peer):
        response = {"round": self.round, "type": "local_model"}

        for attempt in range(1, self.eva_max_retry_attempts + 1):
            if self.model_send_delay is not None:
                await sleep(random.randint(0, self.model_send_delay) / 1000)
            self.logger.info("Participant %d sending round %d local model to peer %s (attempt %d)",
                             self.get_my_participant_index(), self.round, peer, attempt)
            try:
                # TODO this logic is sequential - optimize by having multiple outgoing transfers at once
                res = await self.eva_send_binary(peer, json.dumps(response).encode(), serialize_model(self.model))
                self.logger.info("Local model successfully sent to peer %s", peer)
                break
            except Exception:
                self.logger.exception("Exception when sending model to peer %s", peer)
            attempt += 1

    async def lt_send_local_model(self, peer):
        # We should start seeding this local model
        if not self.torrent_download_manager.is_seeding(self.round, ModelType.LOCAL):
            await self.torrent_download_manager.seed(self.round, ModelType.LOCAL, self.model)

        bencoded_torrent = self.torrent_download_manager.get_torrent_info(
            self.get_my_participant_index(), self.round, ModelType.LOCAL)
        self.send_model_torrent(peer, self.round, ModelType.LOCAL, bencoded_torrent)

    def send_model_torrent(self, peer, round: int, model_type: ModelType, bencoded_torrent: bytes):
        # Send the torrent info to the peer
        # TODO should we have an ack here to improve reliability?
        global_time = self.claim_global_time()
        auth = BinMemberAuthenticationPayload(self.my_peer.public_key.key_to_bin())

        payload = ModelTorrent(round, model_type.value, self.lt_listen_port, bencoded_torrent)
        dist = GlobalTimeDistributionPayload(global_time)
        packet = self._ez_pack(self._prefix, ModelTorrent.msg_id, [auth, dist, payload])
        self.endpoint.send(peer.address, packet)

    @lazy_wrapper(GlobalTimeDistributionPayload, ModelTorrent)
    async def on_model_torrent(self, peer, dist, payload):
        participant_index = self.get_participant_index(peer.public_key.key_to_bin())
        if participant_index == -1:
            self.logger.warning("Received model torrent from peer %s that is not a participant", peer)
            return

        if self.transmission_method != TransmissionMethod.LIBTORRENT:
            self.logger.warning("This peer received a model update but we are not using "
                                "libtorrent as transmission engine")
            return

        model_type = ModelType(payload.model_type)
        self.logger.info("Received %s model torrent from participant %d for round %d",
                         "local" if model_type == ModelType.LOCAL else "aggregated",
                         participant_index, payload.round)
        if self.torrent_download_manager.is_downloading(participant_index, payload.round, model_type):
            self.logger.warning("We are already downloading model with type %d from participant %d for round %d",
                                "local" if model_type == ModelType.LOCAL else "aggregated", participant_index, payload.round)
            return

        # Start to download the torrent
        other_peer_lt_address = (peer.address[0] if not self.is_local_test else "127.0.0.1", payload.lt_port)

        task_name = "download_%d_%d_%d" % (participant_index, payload.round, model_type.value)
        self.register_task(task_name, self.torrent_download_manager.download, participant_index, payload.round, model_type, payload.torrent, other_peer_lt_address).add_done_callback(self.on_model_download_finished)

    def on_model_download_finished(self, task):
        participant, model_round, model_type, serialized_model = task.result()
        self.logger.info("%s model download from participant %d for round %d finished",
                         "Local" if model_type == ModelType.LOCAL else "Aggregated", participant, model_round)
        if model_type == ModelType.LOCAL:
            self.received_local_model(participant, model_round, serialized_model)
        elif model_type == ModelType.AGGREGATED:
            self.received_aggregated_model(participant, model_round, serialized_model)

        # Stop the download
        #self.torrent_download_manager.stop_download(participant, model_round, model_type)

    async def participate_in_round(self):
        """
        Complete a round of training and model aggregation.
        """
        self.is_participating_in_round = True
        self.logger.info("Participant %d starts participating in round %d", self.get_my_participant_index(), self.round)

        # It can happen that this node is still computing the accuracy of the model produced by the previous round
        # when starting the next round. If so, we wait until this accuracy computation is done.
        if self.is_computing_accuracy:
            self.logger.info("Waiting for accuracy computation to finish")
            await self.compute_accuracy_deferred

        # Adopt the aggregated model sent by other nodes
        # TODO there can be inconsistencies in the models received - assume for now they are all the same
        if self.round > 1:
            self.model = random.choice(self.incoming_aggregated_models[self.round - 1])
            self.optimizer = SGDOptimizer(self.model, self.parameters["learning_rate"], self.parameters["momentum"])
            self.incoming_aggregated_models.pop(self.round - 1, None)

        # Train
        epoch_done = await self.train()

        await self.send_local_model()

        avg_model = self.model
        if self.sample_size > 1:
            if self.is_round_representative(self.round) and ((self.round not in self.incoming_local_models) or (self.round in self.incoming_local_models and len(self.incoming_local_models[self.round]) < self.sample_size - 1)):
                await self.round_deferred
                self.logger.info("Round representative %d received %d model(s) from other peers for round %d - "
                                 "starting to average", self.get_my_participant_index(),
                                 len(self.incoming_local_models[self.round]), self.round)

                # Average your model with those of the other participants
                avg_model = self.average_models(self.incoming_local_models[self.round] + [self.model])
                with torch.no_grad():
                    for p, new_p in zip(self.model.parameters(), avg_model.parameters()):
                        p.mul_(0.)
                        p.add_(new_p)

        if self.is_round_representative(self.round):
            if self.round not in self.incoming_aggregated_models:
                self.incoming_aggregated_models[self.round] = []
            self.incoming_aggregated_models[self.round].append(avg_model)
            self.register_task("send_aggregated_model_%d" %
                               self.round, self.send_aggregated_model, self.round, avg_model)

        self.incoming_local_models.pop(self.round, None)
        self.round_deferred = Future()

        self.logger.info("Participant %d finished round %d", self.get_my_participant_index(), self.round)
        if self.round_complete_callback:
            await self.round_complete_callback(self.round, epoch_done)
        self.is_participating_in_round = False

        # Should I participate in the next round again?
        if self.sample_size == 1 and self.is_participant_for_round(self.round + 1):
            self.round += 1
            ensure_future(self.participate_in_round())

        # Check if there is a future round in which we participate and for which we have received all models.
        # If so, start participating in that round.
        for round_nr in self.incoming_aggregated_models:
            if round_nr < self.round:
                continue
            if len(self.incoming_aggregated_models[round_nr]) == 1:
                self.round = round_nr + 1
                ensure_future(self.participate_in_round())

    async def compute_accuracy(self):
        """
        Compute the accuracy/loss of the current model.
        """
        self.logger.info("Computing accuracy of model")
        self.is_computing_accuracy = True
        self.compute_accuracy_deferred = Future()
        correct = example_number = total_loss = num_batches = 0
        with torch.no_grad():
            copied_model = self.model.copy()
            copied_model.eval()
            cur_item = 0
            for data, target in self.dataset.validation_iterator:
                data, target = Variable(data), Variable(target)
                output = copied_model.forward(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                num_batches += 1.0
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                example_number += target.size(0)
                cur_item += 1
                await sleep(0.001)

        accuracy = float(correct) / float(example_number)
        loss = total_loss / float(example_number)
        self.logger.info("Finished computing accuracy of model (accuracy: %f, loss: %f)", accuracy, loss)
        self.dataset.reset_validation_iterator()
        self.is_computing_accuracy = False
        self.compute_accuracy_deferred.set_result(None)
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
        old_model = unserialize_model(old_model_serialized, self.parameters["dataset"], self.parameters["model"])

        # TODO optimize this so we only compare the hash (avoid pulling in the new model)
        new_model_serialized = await self.request_data(other_peer, block.new_model, type=DataType.MODEL)
        new_model = unserialize_model(new_model_serialized, self.parameters["dataset"], self.parameters["model"])

        return self.verify_model_training(old_model, Tensor(datas), torch.LongTensor(targets), new_model)

    def received_local_model(self, participant: int, model_round: int, serialized_model: bytes) -> None:
        self.logger.info("Received local model for round %d from participant %d", model_round, participant)
        if model_round == self.round:
            if not self.is_round_representative(model_round):
                self.logger.warning("We received a local model for round %d from participant %d but we are "
                                    "not the round representative" % model_round, participant)
                return

            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            if model_round not in self.incoming_local_models:
                self.incoming_local_models[model_round] = []
            self.incoming_local_models[model_round].append(incoming_model)
            self.logger.info("Received expected local model (now have %d/%d)",
                             len(self.incoming_local_models[self.round]), self.sample_size - 1)
            if len(self.incoming_local_models[self.round]) == self.sample_size - 1 and not self.round_deferred.done():
                self.round_deferred.set_result(None)
        elif model_round > self.round and self.is_participant_for_round(model_round):
            self.logger.info("Received a local model from participant %d for future round %d",
                             participant, model_round)
            # It is possible that we receive a model for a later round while we are still in an earlier round.
            if model_round not in self.incoming_local_models:
                self.incoming_local_models[model_round] = []
            incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
            self.incoming_local_models[model_round].append(incoming_model)
        else:
            self.logger.warning("Received a local model for a round that is not relevant for us (%d)", model_round)

    def received_aggregated_model(self, participant: int, model_round: int, serialized_model: bytes) -> None:
        if not self.is_participant_for_round(model_round + 1):
            self.logger.warning("Received aggregated model from participant %d for round %d but we are not a "
                                "participant in that round", participant, model_round + 1)

        self.logger.info("Received aggregated model for round %d from participant %d", model_round, participant)
        incoming_model = unserialize_model(serialized_model, self.parameters["dataset"], self.parameters["model"])
        if model_round not in self.incoming_aggregated_models:
            self.incoming_aggregated_models[model_round] = []
        self.incoming_aggregated_models[model_round].append(incoming_model)
        if len(self.incoming_aggregated_models[model_round]) == 1 and not self.is_participating_in_round:
            # Perform this round
            self.round = model_round + 1
            ensure_future(self.participate_in_round())

    def on_receive(self, result: TransferResult):
        self.logger.info(f'Data has been received from peer {result.peer}: {result.info}')
        json_data = json.loads(result.info.decode())
        if "request_id" in json_data:
            # We received this data in response to an earlier request
            if not self.request_cache.has("datarequest", json_data["request_id"]):
                self.logger.warning("Data request cache with ID %d not found!", json_data["request_id"])

            cache = self.request_cache.get("datarequest", json_data["request_id"])
            request_type = DataType(json_data["type"])
            if request_type == DataType.TRAIN_DATA:
                cache.request_future.set_result((json_data["target"], result.data))
            elif request_type == DataType.MODEL:
                cache.request_future.set_result(result.data)
        elif json_data["type"] == "aggregated_model":
            participant = self.get_participant_index(result.peer.public_key.key_to_bin())
            self.received_aggregated_model(participant, json_data["round"], result.data)
        elif json_data["type"] == "local_model":
            participant = self.get_participant_index(result.peer.public_key.key_to_bin())
            self.received_local_model(participant, json_data["round"], result.data)

    def on_send_complete(self, result: TransferResult):
        participant_ind = self.get_participant_index(result.peer.public_key.key_to_bin())
        self.logger.info(f'Outgoing transfer to participant {participant_ind} has completed: {result.info}')

    def on_error(self, peer, exception):
        self.logger.error(f'An error has occurred in transfer to peer {peer}: {exception}')
