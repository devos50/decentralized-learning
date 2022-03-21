from accdfl.core.blocks import ModelUpdateBlock


class ModelUpdateBlockListener:
    BLOCK_CLASS = ModelUpdateBlock

    def received_block(self, block):
        pass
