from abc import ABC
from typing import Tuple

import numpy as np
import pandas as pd


class Predictor(ABC):
    """Abstract Base Class for Predictors"""

    @staticmethod
    def training_test_split(data: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx = int(len(data) * ratio)
        training_data, test_data = data[:idx], data[idx:]

        return training_data, test_data

    @staticmethod
    def rolling_window(data: pd.DataFrame, size: int, offset_left: int = 0, offset_right: int = 0) -> np.ndarray:
        matrix = data.values
        window_starts = range(offset_left, matrix.shape[0] - (size - 1) + offset_right)
        return np.array([matrix[i:i + size] for i in window_starts])
