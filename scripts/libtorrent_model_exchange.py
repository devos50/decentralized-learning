"""
Start two libtorrent sessions. One session seeds a model and the other session downloads the model.
"""
import sys
import time

import libtorrent as lt

# Seeder
seeder_ses = lt.session()
seeder_ses.listen_on(6871, 6881)

# Downloader
download_ses = lt.session()
download_ses.listen_on(6881, 6891)
h = lt.add_magnet_uri(download_ses, "magnet:?xt=urn:btih:648bee814b508e05c46fe13869c4d71dfe5c27b4", {'save_path': './'})

while True:
    s = h.status()
    state_str = ['queued', 'checking', 'downloading metadata',
                 'downloading', 'finished', 'seeding', 'allocating', 'checking fastresume']
    print('\r%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' % \
    (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
     s.num_peers, state_str[s.state]))
    sys.stdout.flush()
    time.sleep(1)
