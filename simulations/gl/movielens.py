from asyncio import ensure_future

from simulations.args import get_args
from simulations.gl.gl_simulation import GLSimulation

if __name__ == "__main__":
    args = get_args("movielens", default_lr=0.2)
    simulation = GLSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
