from accdfl.gl.community import GLCommunity


class SimulatedGLCommunity(GLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.my_peer.add_address(self.endpoint.wan_address)