from asyncio import get_event_loop
from logging import LoggerAdapter


class SimulationLoggerAdapter(LoggerAdapter):

    def process(self, msg, kwargs):
        return ' [t=%d] %s' % (get_event_loop().time(), msg), kwargs
