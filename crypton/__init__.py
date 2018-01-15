__version__ = '0.0.2'

APP_NAME = 'Crypton'

# pylint: disable=all

import os  # isort:skip
ENV = os.getenv(f'{APP_NAME.upper()}_ENV', 'config')  # isort:skip

import kick  # isort:skip
kick.start(f'{APP_NAME.lower()}', config_variant=ENV)  # isort:skip

from kick import config, logger  # isort:skip

import asyncio
from pathlib import Path

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

CACHE_DIR = Path.home() / '.cache' / APP_NAME
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)

from .crypton import Crypton  # isort:skip
from .predictor import ProphetPredictor  # isort:skip
