import os
from asyncio import get_event_loop
from binascii import hexlify

from accdfl.core.session_settings import DFLSettings, LearningSettings, SessionSettings
from ipv8.configuration import ConfigBuilder

from simulations.learning_simulation import LearningSimulation
from simulations.settings import SimulationSettings


class DFLSimulation(LearningSimulation):

    def __init__(self, settings: SimulationSettings) -> None:
        super().__init__(settings)
        self.latest_accuracy_check_round: int = 0

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.settings.bypass_model_transfers:
            builder.add_overlay("DFLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("DLFCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        peer_pk = None
        lowest_latency_peer_id = -1
        if self.settings.fix_aggregator:
            lowest_latency_peer_id = self.determine_peer_with_lowest_median_latency()
            peer_pk = self.nodes[lowest_latency_peer_id].overlays[0].my_peer.public_key.key_to_bin()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.settings.learning_rate,
            momentum=self.settings.momentum,
            batch_size=self.settings.batch_size
        )

        dfl_settings = DFLSettings(
            sample_size=self.settings.sample_size,
            num_aggregators=self.settings.num_aggregators,
            success_fraction=1.0,
            aggregation_timeout=2.0,
            ping_timeout=5,
            inactivity_threshold=1000,
            fixed_aggregator=peer_pk if self.settings.fix_aggregator else None
        )

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.settings.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=len(self.nodes),
            dfl=dfl_settings,
            model=self.settings.model,
            alpha=self.settings.alpha,
            partitioner=self.settings.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.settings.train_device_name,
        )

        for ind, node in enumerate(self.nodes):
            node.overlays[0].aggregate_complete_callback = lambda round_nr, model, i=ind: self.on_aggregate_complete(i, round_nr, model)
            node.overlays[0].setup(self.session_settings)

        # If we fix the aggregator, we assume unlimited upload/download slots
        if self.settings.fix_aggregator:
            print("Overriding max. EVA transfers for peer %d" % lowest_latency_peer_id)
            self.nodes[lowest_latency_peer_id].overlays[0].eva.settings.max_simultaneous_transfers = 100000

        if self.settings.bypass_model_transfers:
            # Inject the nodes in each community
            for node in self.nodes:
                node.overlays[0].nodes = self.nodes

    async def on_aggregate_complete(self, ind: int, round_nr: int, model):
        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        print("Round %d completed - bytes up: %d, bytes down: %d" % (round_nr, tot_up, tot_down))

        if round_nr % self.settings.accuracy_logging_interval == 0 and round_nr > self.latest_accuracy_check_round:
            print("Will compute accuracy for round %d!" % round_nr)
            accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.settings.accuracy_device_name)
            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                group = "\"s=%d, a=%d\"" % (self.settings.sample_size, self.settings.num_aggregators)
                out_file.write("%s,%s,%f,%d,%d,%f,%f\n" % (self.settings.dataset, group, get_event_loop().time(),
                                                           ind, round_nr, accuracy, loss))
            self.latest_accuracy_check_round = round_nr

        if self.settings.num_rounds and round_nr >= self.settings.num_rounds:
            self.on_simulation_finished()
            self.loop.stop()

    def on_simulation_finished(self) -> None:
        super().on_simulation_finished()

        # Write away the view histories
        with open(os.path.join(self.data_dir, "view_histories.csv"), "w") as out_file:
            out_file.write("peer,update_time,peers\n")
            for peer_id, node in enumerate(self.nodes):
                for update_time, active_peers in node.overlays[0].active_peers_history:
                    active_peers_str = "-".join(active_peers)
                    out_file.write("%d,%f,%s\n" % (peer_id + 1, update_time, active_peers_str))

        # Write the determine sample durations
        with open(os.path.join(self.data_dir, "determine_sample_durations.csv"), "w") as out_file:
            out_file.write("peer,start_time,end_time\n")
            for peer_id, node in enumerate(self.nodes):
                for start_time, end_time in node.overlays[0].determine_sample_durations:
                    out_file.write("%d,%f,%f\n" % (peer_id + 1, start_time, end_time))

        # Write away the outgoing bytes statistics
        with open(os.path.join(self.data_dir, "outgoing_bytes_statistics.csv"), "w") as bw_file:
            bw_file.write("peer,lm_model_bytes,lm_midas_bytes,ping_bytes,pong_bytes\n")
            for ind, node in enumerate(self.nodes):
                bw_file.write("%d,%d,%d,%d,%d\n" % (ind + 1,
                                                    node.overlays[0].bandwidth_statistics["lm_model_bytes"],
                                                    node.overlays[0].bandwidth_statistics["lm_midas_bytes"],
                                                    node.overlays[0].bandwidth_statistics["ping_bytes"],
                                                    node.overlays[0].bandwidth_statistics["pong_bytes"]))
