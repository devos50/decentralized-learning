from asyncio import ensure_future
from binascii import hexlify

from accdfl.core.session_settings import SessionSettings, LearningSettings, GLSettings
from ipv8.configuration import ConfigBuilder

from simulations.settings import SimulationSettings
from simulations.simulation import DLSimulation


class BasicGLSimulation(DLSimulation):

    def get_session_settings(self) -> SessionSettings:
        learning_settings = LearningSettings(
            learning_rate=self.settings.learning_rate,
            momentum=self.settings.momentum,
            batch_size=self.settings.batch_size
        )

        gl_settings = GLSettings(
            round_duration=10
        )

        return SessionSettings(
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            target_participants=len(self.nodes),
            dataset="cifar10",
            learning=learning_settings,
            work_dir=self.data_dir,
            gl=gl_settings,
        )

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        builder.add_overlay("SimulatedGLCommunity", "my peer", [], [], {}, [])
        return builder


if __name__ == "__main__":
    settings = SimulationSettings()
    simulation = BasicGLSimulation(settings)
    ensure_future(simulation.run())

    simulation.loop.run_forever()
