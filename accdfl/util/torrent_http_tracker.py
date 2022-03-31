import json
import logging

from aiohttp import web

from libtorrent import bencode


HTTP_BAD_REQUEST = 400


class RESTResponse(web.Response):

    def __init__(self, body=None, headers=None, content_type=None, status=200, **kwargs):
        if not isinstance(status, int):
            status = getattr(status, 'status_code')
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
            content_type = 'application/json'
        super().__init__(body=body, headers=headers, content_type=content_type, status=status, **kwargs)


class RESTEndpoint:

    def __init__(self, middlewares=()):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.app = web.Application(middlewares=middlewares, client_max_size=2*1024**2)
        self.endpoints = {}
        self.setup_routes()

    def setup_routes(self):
        pass

    def add_endpoint(self, prefix, endpoint):
        self.endpoints[prefix] = endpoint
        self.app.add_subapp(prefix, endpoint.app)


class TorrentHTTPTracker:

    def __init__(self, port):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.listening_port = None
        self.site = None
        self.port = port
        self.peers = {}

    async def start(self):
        """
        Start the HTTP Tracker
        """
        app = web.Application()
        app.add_routes([web.get('/announce', self.handle_announce_request)])
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()

        attempts = 0
        while attempts < 20:
            try:
                self.site = web.TCPSite(runner, '0.0.0.0', self.port)
                await self.site.start()
                break
            except OSError:
                attempts += 1
                self.port += 1

    async def stop(self):
        """
        Stop the HTTP Tracker, returns a deferred that fires when the server is closed.
        """
        if self.site:
            return await self.site.stop()

    async def handle_announce_request(self, request):
        """
        Return a bencoded dictionary with peers.
        """
        if 'info_hash' not in request.query:
            return RESTResponse("infohash argument missing", status=101)

        infohash = request.query["info_hash"]
        peer_id = request.query["peer_id"]
        if infohash not in self.peers:
            self.peers[infohash] = {}

        if request.query["event"] == "stopped" and peer_id in self.peers[infohash]:
            self.peers[infohash].pop(peer_id)
        else:
            self.peers[infohash][peer_id] = {"id": peer_id, "ip": request.remote, "port": int(request.query["port"])}

        response_dict = {
            "interval": 5,
            "peers": list(self.peers[infohash].values())
        }

        return RESTResponse(bencode(response_dict))
