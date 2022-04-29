from enum import Enum


class TransmissionMethod(Enum):
    EVA = 0


class NodeDelta(Enum):
    JOIN = 0
    LEAVE = 1
