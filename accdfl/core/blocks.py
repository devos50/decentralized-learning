from binascii import unhexlify

from accdfl.trustchain.block import TrustChainBlock


class ModelUpdateBlock(TrustChainBlock):

    def __init__(self, *args, **kwargs):
        super(ModelUpdateBlock, self).__init__(*args, **kwargs)
        self.inputs = []
        self.old_model = self.new_model = None
        if self.transaction:
            self.old_model = unhexlify(self.transaction["old_model"])
            self.new_model = unhexlify(self.transaction["new_model"])
