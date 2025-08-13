from asyncio import ensure_future

from simulations.args import get_args
from simulations.teleportation.teleportation_simulation import TeleportationSimulation

if __name__ == "__main__":
    args = get_args("emotion", default_lr=0.0005, default_momentum=0)
    simulation = TeleportationSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
