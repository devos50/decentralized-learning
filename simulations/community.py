from accdfl.core.community import DFLCommunity
from accdfl.core.model import serialize_model


class SimulatedDFLCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None

    async def send_aggregated_model(self, round, model):
        """
        Inject the aggregated model directly in the node.
        """
        serialized_model = serialize_model(model)
        participants_next_round = self.get_participants_for_round(round + 1)
        for participant_ind in participants_next_round:
            if participant_ind == self.get_my_participant_index():
                continue

            self.logger.info("Participant %d sending round %d aggregated model to participant %d",
                             self.get_my_participant_index(), round, participant_ind)

            target_node = self.nodes[participant_ind]
            target_node.overlays[0].received_aggregated_model(self.get_my_participant_index(), round, serialized_model)

    async def send_local_model(self):
        if self.is_round_representative(self.round):
            return

        round_representative = self.get_round_representative(self.round)
        serialized_model = serialize_model(self.model)
        target_node = self.nodes[round_representative]
        target_node.overlays[0].received_local_model(self.get_my_participant_index(), self.round, serialized_model)