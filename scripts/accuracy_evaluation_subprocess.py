import concurrent.futures
import os
from asyncio import get_event_loop, ensure_future

from accdfl.core.evaluator import Evaluator
from accdfl.core.model import create_model

evaluator = None


def init_globals(parameters):
    global evaluator
    evaluator = Evaluator(os.path.join(os.environ["HOME"], "dfl-data"), parameters)


def evaluate_accuracy(model):
    return evaluator.evaluate_accuracy(model)


async def main():
    parameters = {
        "batch_size": 200,
        "dataset": "cifar10",
        "model": "gnlenet",
    }

    # Create a model to test
    model = create_model(parameters["dataset"], parameters["model"])

    executor = concurrent.futures.ProcessPoolExecutor(initializer=init_globals, initargs=(parameters,), max_workers=1)
    accuracy, loss = await get_event_loop().run_in_executor(executor, evaluate_accuracy, model)
    print("Accuracy: %f, loss: %f" % (accuracy, loss))
    get_event_loop().stop()


if __name__ == "__main__":
    ensure_future(main())
    get_event_loop().run_forever()