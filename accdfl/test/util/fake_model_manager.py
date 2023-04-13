from asyncio import sleep

from accdfl.core.model_manager import ModelManager


class FakeModelManager(ModelManager):
    """
    A model manager that does not actually train the model but simply sleeps.
    """
    train_time = 0.001

    async def train(self) -> int:
        await sleep(self.train_time)
        return 0
