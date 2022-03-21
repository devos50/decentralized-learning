import logging
from asyncio import Future

from ipv8.requestcache import RandomNumberCache


class DataRequestCache(RandomNumberCache):
    """
    This request cache keeps track of outstanding crawl requests.
    """

    CRAWL_TIMEOUT = 30.0

    def __init__(self, community: "DFLCommunity", request_future: Future) -> None:
        super(DataRequestCache, self).__init__(community.request_cache, "datarequest")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.community = community
        self.request_future = request_future

    @property
    def timeout_delay(self) -> float:
        return DataRequestCache.CRAWL_TIMEOUT

    def received_not_found_response(self) -> None:
        self.community.request_cache.pop("datarequest", self.number)
        self.request_future.set_result(None)

    def on_timeout(self) -> None:
        self._logger.info("Timeout for data request with id %d", self.number)
        self.request_future.set_result(None)
