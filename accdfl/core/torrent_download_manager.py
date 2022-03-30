import logging
import os
from asyncio import sleep

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
        self.session = lt.session()
        self.model_downloads = {}

    def start(self, listen_port: int) -> None:
        self.logger.info("Starting libtorrent session, listening on port %d", listen_port)
        self.session.listen_on(listen_port, listen_port + 5)

    def is_seeding(self, round: int, model_type: ModelType) -> bool:
        if (round, model_type) in self.model_downloads:
            download = self.model_downloads[(round, model_type)]
            if download.status().state == 5:  # Seeding
                return True
        return False

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
        torrent = lt.bdecode(create_torrent_file(model_file_path))
        torrent_info = lt.torrent_info(torrent)
        seed_torrent_info = {
            "ti": torrent_info,
            "save_path": self.data_dir
        }
        upload = self.session.add_torrent(seed_torrent_info)
        self.model_downloads[(round, model_type)] = upload
        for _ in range(50):
            await sleep(0.2)
            if upload.status().state == 5:
                return succeed(None)

        return fail(RuntimeError("Torrent not seeding after 10 seconds!"))
