"""
Train n standalone Movielens models for a specified number of rounds.
"""
import asyncio
import logging

from scripts.run import get_args, run

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(get_args(default_lr=0.25), "movielens"))
