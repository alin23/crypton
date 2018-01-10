import matplotlib

from .. import logger

try:
    import matplotlib.pyplot as plt
except:  # pylint: disable=bare-except
    GUI_ENV = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
    for gui in GUI_ENV:
        try:
            logger.debug(f'Testing {gui}')
            matplotlib.use(gui, warn=False, force=True)
            import matplotlib.pyplot as plt
            break
        except:  # pylint: disable=bare-except
            continue
    logger.debug(f'Using: {matplotlib.get_backend()}')

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use(['fivethirtyeight'])

# pylint: disable=wrong-import-position
from .prophet import ProphetPredictor  # isort:skip
from .keras import KerasPredictor  # isort:skip
from .base import Predictor  # isort:skip
