from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
# pylint: disable=no-name-in-module,import-error
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.layers.noise import AlphaDropout

from . import plt
from .. import config, logger
from .base import Predictor
from ..types import CurrencyPair


class KerasPredictor(Predictor):
    def __init__(
        self,
        pair: CurrencyPair,
        regressors: List[str] = None,
        x_window_size: int = 10,
        y_window_size: int = 5,
    ) -> None:
        super().__init__()
        self.currency, self.to_currency = pair
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.regressors = regressors or []
        self.init_model()

    def init_model(self, features: int = None, batch_size: int = None):
        features = len(self.regressors) or features or config.keras.features
        batch_size = batch_size or config.keras.batch_size

        inp = Input(shape=(self.x_window_size, features), batch_shape=(batch_size, self.x_window_size, features))
        lstm = LSTM(6, activation='selu', kernel_initializer='lecun_normal', stateful=True, unroll=True)(inp)
        dropout = AlphaDropout(0.4)(lstm)
        output = Dense(self.y_window_size)(dropout)
        self.model = Model(inputs=inp, outputs=output)
        self.model.compile(loss='mean_squared_logarithmic_error', optimizer='nadam', metrics=['accuracy'])

    def format_data(self, data: pd.DataFrame, y_column: str, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        data = data.dropna()
        x = data[self.regressors] if self.regressors else data  # pylint: disable=invalid-name
        y = data[y_column]  # pylint: disable=invalid-name

        inputs = self.rolling_window(x, self.x_window_size, offset_right=-self.y_window_size)
        outputs = self.rolling_window(y, self.y_window_size, offset_left=self.x_window_size)

        if batch_size:
            samples = (inputs.shape[0] // batch_size) * batch_size
            inputs = inputs[-samples:]
            outputs = outputs[-samples:]

        return inputs, outputs

    def learn(
        self,
        data: pd.DataFrame,
        y_column: str,
        test_data: pd.DataFrame = None,
        epochs: int = None,
        batch_size: int = None
    ) -> None:
        logger.debug('Learning from data...')

        epochs = epochs or config.keras.epochs
        batch_size = batch_size or config.keras.batch_size
        inputs, outputs = self.format_data(data, y_column, batch_size=batch_size)
        if test_data is not None:
            test_data = self.format_data(test_data, y_column, batch_size=batch_size)

        self.model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=test_data)

    def predict(self, data: pd.DataFrame, y_column: str, batch_size: int = None) -> np.ndarray:
        logger.debug('Forecasting...')

        batch_size = batch_size or config.keras.batch_size
        inputs, _ = self.format_data(data, y_column, batch_size=batch_size)
        return self.model.predict(inputs, batch_size=batch_size)

    def plot(
        self, forecast: np.ndarray, training_data: pd.Series, test_data: pd.Series = None, show: bool = False
    ) -> matplotlib.figure.Figure:
        logger.debug('Plotting...')

        history = training_data
        timeframe = history.index[-1] - history.index[-2]
        forecast = pd.Series(
            forecast, index=pd.date_range(start=history.index[-1] + timeframe, periods=len(forecast), freq=timeframe)
        )

        highest_datapoint = max(history.max(), forecast.max())
        lowest_datapoint = min(history.min(), forecast.min())
        if test_data is not None:
            highest_datapoint = max(highest_datapoint, test_data.max())
            lowest_datapoint = min(lowest_datapoint, test_data.min())

        fig, ax1 = plt.subplots()
        # ax1.set_ylim(bottom=lowest_datapoint - (lowest_datapoint / 2), top=highest_datapoint + (highest_datapoint / 2))

        ax1.plot(history, color='red', linewidth=config.plot.linewidth)
        if test_data is not None:
            ax1.plot(test_data, color='orange', linewidth=config.plot.linewidth)

        ax1.plot(forecast, color='black', linestyle=':', linewidth=config.plot.linewidth + 0.2)
        ax1.set_title(
            f'{self.currency}/{self.to_currency} Price (Orange) vs {self.currency}/{self.to_currency} Price Forecast (Black)'
        )
        ax1.set_ylabel(f'{self.currency}/{self.to_currency} Price')
        ax1.set_xlabel('Date')

        legend = ax1.legend()

        texts = legend.get_texts()
        if test_data is not None:
            texts[0].set_text('Actual Price (training data)')
            texts[1].set_text('Actual Price (test data)')
            if len(texts) == 3:
                texts[2].set_text('Forecasted Price')
        else:
            texts[0].set_text('Actual Price')
            texts[1].set_text('Forecasted Price')

        if show:
            plt.show()

        return fig

    def test_prediction(
        self,
        data: pd.DataFrame,
        y_column: str,
        training_test_ratio: float,
        epochs: int = config.keras.epochs,
        batch_size: int = config.keras.batch_size
    ) -> None:
        self.init_model(features=data.shape[1], batch_size=batch_size)

        training_data, test_data = self.training_test_split(data, training_test_ratio)

        self.learn(training_data, y_column, batch_size=batch_size, epochs=epochs, test_data=test_data)
        forecast = self.predict(test_data, y_column, batch_size=batch_size)[:, 0]
        forecast = forecast[~np.isnan(forecast)]

        if len(forecast) > 0:
            self.plot(forecast, training_data=training_data[y_column], test_data=test_data[y_column], show=True)
