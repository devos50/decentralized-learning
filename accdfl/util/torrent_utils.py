import os

import libtorrent as lt


def create_torrent_file(file_path) -> bytes:
    fs = lt.file_storage()
    fs.add_file(os.path.basename(file_path), os.path.getsize(file_path))
    flags = lt.create_torrent_flags_t.optimize
    torrent = lt.create_torrent(fs, flags=flags)
    lt.set_piece_hashes(torrent, os.path.dirname(file_path))
    t1 = torrent.generate()
    return lt.bencode(t1)
