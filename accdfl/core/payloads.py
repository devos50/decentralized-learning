from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=10)
class AdvertiseMembership:
    round: int
