from asyncio import ensure_future

from simulations.args import get_args
from simulations.diloco.diloco_simulation import DiLoCoSimulation

if __name__ == "__main__":
    args = get_args("cifar10")
    simulation = DiLoCoSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
