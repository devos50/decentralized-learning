from enum import Enum


class TransmissionMethod(Enum):
    EVA = 0


class NodeMembershipChange(Enum):
    JOIN = 0
    LEAVE = 1
