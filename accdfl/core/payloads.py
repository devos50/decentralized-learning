from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=10)
class AdvertiseMembership:
    round: int
    change: int


@dataclass(msg_id=11)
class PingPayload:
    identifier: int


@dataclass(msg_id=12)
class PongPayload:
    identifier: int
