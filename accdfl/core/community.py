from binascii import unhexlify

from accdfl.util.eva_protocol import EVAProtocolMixin

from ipv8.community import Community


class DFLCommunity(EVAProtocolMixin, Community):
    community_id = unhexlify('d5889074c1e4c60423cdb6e9307aa0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eva_init()

        self.eva_register_receive_callback(self.on_receive)
        self.eva_register_send_complete_callback(self.on_send_complete)
        self.eva_register_error_callback(self.on_error)

    def on_receive(self, peer, binary_info, binary_data, nonce):
        print("RECCCC")
        self.logger.info(f'Data has been received: {binary_info}')

    def on_send_complete(self, peer, binary_info, binary_data, nonce):
        self.logger.info(f'Transfer has been completed: {binary_info}')

    def on_error(self, peer, exception):
        self.logger.error(f'Error has been occurred: {exception}')
