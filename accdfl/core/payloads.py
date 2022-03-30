from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=10)
class DataRequest:
    request_id: int
    data_hash: bytes
    request_type: int


@dataclass(msg_id=11)
class DataNotFoundResponse:
    request_id: int


@dataclass(msg_id=12)
class ModelTorrent:
    round: int
    model_type: int
    lt_port: int
    torrent: bytes  # Torrent in bencoded form
