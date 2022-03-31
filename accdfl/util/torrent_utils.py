import os
from typing import List

import libtorrent as lt


def create_torrent_file(file_path, trackers: List[str] = None) -> bytes:
    fs = lt.file_storage()
    fs.add_file(os.path.basename(file_path), os.path.getsize(file_path))
    flags = lt.create_torrent_flags_t.optimize
    torrent = lt.create_torrent(fs, flags=flags)
    if trackers:
        for tracker_url in trackers:
            torrent.add_tracker(tracker_url)
    lt.set_piece_hashes(torrent, os.path.dirname(file_path))
    t1 = torrent.generate()
    return lt.bencode(t1)
