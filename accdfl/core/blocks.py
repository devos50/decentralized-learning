from binascii import unhexlify

from accdfl.trustchain.block import TrustChainBlock


class ModelUpdateBlock(TrustChainBlock):

    def __init__(self, *args, **kwargs):
        super(ModelUpdateBlock, self).__init__(*args, **kwargs)
        self.inputs = []
        if self.transaction:
            self.inputs = [unhexlify(data_hash) for data_hash in self.transaction["inputs"]]
