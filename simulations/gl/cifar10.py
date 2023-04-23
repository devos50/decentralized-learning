from asyncio import ensure_future

from simulations.args import get_args
from simulations.gl.gl_simulation import GLSimulation

if __name__ == "__main__":
    args = get_args("cifar10", default_lr=0.002, default_momentum=0.9)
    simulation = GLSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
