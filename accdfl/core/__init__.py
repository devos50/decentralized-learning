from enum import Enum


class TransmissionMethod(Enum):
    EVA = 0


class NodeMembershipChange(Enum):
    JOIN = 0
    LEAVE = 1


class State(Enum):
    IDLE = 0
    TRAINING = 1
    AGGREGATING = 2
