"""
Train n standalone Movielens models for a specified number of rounds.
"""
import asyncio
import logging
import os

from accdfl.core.session_settings import LearningSettings
from scripts.run import run

learning_settings = LearningSettings(
    learning_rate=0.25 if "LEARNING_RATE" not in os.environ else float(os.environ["LEARNING_RATE"]),
    momentum=0,
    batch_size=20
)

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(learning_settings, "movielens"))
