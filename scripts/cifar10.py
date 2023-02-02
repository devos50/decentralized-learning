"""
Train n standalone CIFAR10 models for a specified number of rounds.
"""
import asyncio
import logging
import os

from accdfl.core.session_settings import LearningSettings
from scripts.run import run

learning_settings = LearningSettings(
    learning_rate=0.001 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
    momentum=0.9,
    batch_size=200
)

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(learning_settings, "cifar10"))
