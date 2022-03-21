from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=10)
class DataRequest:
    request_id: int
    data_hash: bytes


@dataclass(msg_id=11)
class DataNotFoundResponse:
    request_id: int
