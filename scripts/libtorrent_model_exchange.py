"""
Start two libtorrent sessions. One session seeds a model and the other session downloads the model.
"""
import os
import sys
import time

import libtorrent as lt

from accdfl.util.torrent_utils import create_torrent_file

MODEL_FILE_DIR = os.path.abspath("../data")
MODEL_FILE_PATH = os.path.join(MODEL_FILE_DIR, "local_model_3.dat")
DOWNLOAD_DIR = os.path.abspath("../data/download")

# Setup a (fake) model to be seeded and exchange
with open(MODEL_FILE_PATH, "w") as out_file:
    out_file.write("a" * 1000)

# Create the torrent to be seeded
torrent = lt.bdecode(create_torrent_file(MODEL_FILE_PATH))
torrent_info = lt.torrent_info(torrent)
print(torrent)

# Setup downloader
download_ses = lt.session()
download_ses.listen_on(6881, 6891)

# Setup the seeder session and start seeding the model
seeder_ses = lt.session()
seeder_ses.listen_on(6871, 6881)
seed_torrent_info = {
    "ti": torrent_info,
    "save_path": MODEL_FILE_DIR
}
h = seeder_ses.add_torrent(seed_torrent_info)

while True:
    s = h.status()
    state_str = ['queued', 'checking', 'downloading metadata',
                 'downloading', 'finished', 'seeding', 'allocating', 'checking fastresume']
    print('\r%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' % \
    (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
     s.num_peers, state_str[s.state]))
    sys.stdout.flush()
    if s.state == 5:
        break
    time.sleep(1)

# Start download
download_torrent_info = {
    "ti": torrent_info,
    "save_path": DOWNLOAD_DIR
}

print("Will start download")
download = download_ses.add_torrent(download_torrent_info)
while True:
    s = download.status()
    print('\r%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' % \
          (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
           s.num_peers, state_str[s.state]))
    sys.stdout.flush()
    time.sleep(1)