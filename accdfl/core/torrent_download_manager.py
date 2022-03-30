import logging
import os
from asyncio import sleep, CancelledError
from typing import Optional, Tuple

import libtorrent as lt

import torch.nn as nn

from accdfl.core.model import ModelType, serialize_model
from accdfl.util.torrent_utils import create_torrent_file
from ipv8.util import succeed, fail


class TorrentDownloadManager:
    """
    This manager manages the libtorrent model seeding and downloading.
    """

    def __init__(self, data_dir: str, participant_index: int):
        self.data_dir = data_dir
        self.participant_index = participant_index
        self.logger = logging.getLogger(__name__)
        self.model_downloads = {}
        self.model_torrents = {}

        settings = {
            "enable_upnp": False,
            "enable_dht": False,
            "enable_lsd": False,
            "enable_natpmp": False,
            "allow_multiple_connections_per_ip": True
        }
        self.session = lt.session(settings)

    def start(self, listen_port: int) -> None:
        self.logger.info("Starting libtorrent session, listening on port %d", listen_port)
        self.session.listen_on(listen_port, listen_port + 5)

    def get_torrent_info(self, participant_index: int, round: int, model_type: ModelType) -> Optional[bytes]:
        if (participant_index, round, model_type) in self.model_torrents:
            return self.model_torrents[(participant_index, round, model_type)]
        return None

    def is_seeding(self, round: int, model_type: ModelType) -> bool:
        if (self.participant_index, round, model_type) in self.model_downloads:
            download = self.model_downloads[(self.participant_index, round, model_type)]
            if download.status().state == 5:  # Seeding
                return True
        return False

    def is_downloading(self, participant_index: int, round: int, model_type: ModelType) -> bool:
        return (participant_index, round, model_type) in self.model_downloads

    async def seed(self, round: int, model_type: ModelType, model: nn.Module):
        """
        Start seeding a given model if it is not seeding already.
        """
        if self.is_seeding(round, model_type):
            return succeed(None)

        # Serialize the model and store it in the data directory.
        model_name = "%d_%d_%s" % (self.participant_index, round, "local" if model_type == ModelType.LOCAL else "aggregated")
        model_file_path = os.path.join(self.data_dir, model_name)
        with open(model_file_path, "wb") as model_file:
            model_file.write(serialize_model(model))

        # Create a torrent and start seeding the model
        bencoded_torrent = create_torrent_file(model_file_path)
        self.model_torrents[(self.participant_index, round, model_type)] = bencoded_torrent
        torrent = lt.bdecode(bencoded_torrent)
        torrent_info = lt.torrent_info(torrent)
        seed_torrent_info = {
            "ti": torrent_info,
            "save_path": self.data_dir
        }
        upload = self.session.add_torrent(seed_torrent_info)
        self.model_downloads[(self.participant_index, round, model_type)] = upload
        for _ in range(100):
            await sleep(0.1)
            if upload.status().state == 5:
                return succeed(None)

        return fail(RuntimeError("Torrent not seeding after 10 seconds!"))

    async def download(self, participant_index: int, round: int, model_type: ModelType, bencoded_torrent: bytes,
                       other_peer_lt_address: Tuple):
        self.model_torrents[(participant_index, round, model_type)] = bencoded_torrent
        torrent_info = lt.torrent_info(lt.bdecode(bencoded_torrent))
        download_torrent_info = {
            "ti": torrent_info,
            "save_path": self.data_dir
        }

        download = self.session.add_torrent(download_torrent_info)
        self.model_downloads[(participant_index, round, model_type)] = download
        try:
            await sleep(1)
        except CancelledError:
            self.logger.warning("Ignoring cancellation of download task")
        download.connect_peer(other_peer_lt_address)

        while True:
            s = download.status()
            state_str = ['queued', 'checking', 'downloading metadata',
                         'downloading', 'finished', 'seeding', 'allocating', 'checking fastresume']
            self.logger.debug('%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' %
                              (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000,
                               s.num_peers, state_str[s.state]))
            if s.state == 4 or s.state == 5:
                # The download seems to be finished
                model_name = "%d_%d_%s" % (self.participant_index, round, "local" if model_type == ModelType.LOCAL else "aggregated")
                with open(os.path.join(self.data_dir, model_name), "rb") as model_file:
                    serialized_model = model_file.read()
                return participant_index, round, model_type, serialized_model
            await sleep(0.2)
