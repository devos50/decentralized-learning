from typing import Callable, Coroutine

from ipv8.types import Peer

from accdfl.util.eva.exceptions import TransferException
from accdfl.util.eva.result import TransferResult

TransferCompleteCallback = Callable[[TransferResult], Coroutine]
TransferErrorCallback = Callable[[Peer, TransferException], Coroutine]
