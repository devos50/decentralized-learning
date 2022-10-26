from typing import Iterator, Tuple, List

import networkx as nx
import numpy as np


def ExponentialTwoGraph(size: int) -> nx.DiGraph:
    """Generate graph topology such that each points only
    connected to a point such that the index difference is the power of 2.
    Example: A ExponentialTwoGraph with 12 nodes:
    .. plot::
        :context: close-figs
        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.ExponentialTwoGraph(12)
        >>> nx.draw_circular(G)
    """
    assert size > 0
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def GetDynamicOnePeerSendRecvRanks(
        topo: nx.DiGraph, self_rank: int) -> Iterator[Tuple[List[int], List[int]]]:
    """A utility function to generate 1-outoging send rank and corresponding recieving rank(s).
    Args:
        topo (nx.DiGraph): The base topology to generate dynamic send and receive ranks.
        self_rank (int): The self rank.
    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_ranks, recv_ranks.
    Example:
        >>> from bluefog.common import topology_util
        >>> topo = topology_util.PowerTwoRingGraph(10)
        >>> gen = topology_util.GetDynamicOnePeerSendRecvRanks(topo, 0)
        >>> for _ in range(10):
        >>>     print(next(gen))
    """
    # Generate all outgoing ranks sorted by clock-wise. (Imagine all ranks put on a clock.)
    size = topo.number_of_nodes()
    sorted_send_ranks = []
    for rank in range(size):
        sorted_ranks = sorted(topo.successors(rank),
                              key=lambda r, rk=rank: r-rk if r >= rk else r-rk+size)
        if sorted_ranks[0] == rank:
            sorted_ranks = sorted_ranks[1:]  # remove the self-loop
        sorted_send_ranks.append(sorted_ranks)

    self_degree = topo.out_degree(self_rank) - 1
    index = 0
    while True:
        send_rank = sorted_send_ranks[self_rank][index % self_degree]
        recv_ranks = []
        for other_rank in range(size):
            if other_rank == self_rank:
                continue
            degree = topo.out_degree(other_rank) - 1
            if sorted_send_ranks[other_rank][index % degree] == self_rank:
                recv_ranks.append(other_rank)

        yield [send_rank], recv_ranks
        index += 1
