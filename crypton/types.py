from enum import IntEnum
from typing import Tuple, Union
from datetime import date, datetime

import pandas as pd


class TimeFrame(IntEnum):
    MINUTE = 60
    HOUR = 60 * 60
    DAY = 60 * 60 * 24

    @classmethod
    def get(cls, timeframe):
        if isinstance(timeframe, str):
            return cls[timeframe.upper()]
        return cls(timeframe)


# pylint: disable=invalid-name
CurrencyPair = Tuple[str, str]
Frequency = Union[TimeFrame, str, pd.DateOffset]
Date = Union[str, datetime, date]
