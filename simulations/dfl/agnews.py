from asyncio import ensure_future

from simulations.args import get_args
from simulations.dfl.dfl_simulation import DFLSimulation

if __name__ == "__main__":
    args = get_args("ag_news", default_lr=0.00005, default_momentum=0)
    simulation = DFLSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
