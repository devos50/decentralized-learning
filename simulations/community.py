from accdfl.core.community import DFLCommunity


class SimulatedDFLCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
