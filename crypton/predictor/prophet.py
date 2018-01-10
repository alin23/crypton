import time
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
from fbprophet import Prophet

from . import plt
from .. import config, logger
from .base import Predictor
from ..types import CurrencyPair


class ProphetPredictor(Predictor):
    def __init__(self, pair: CurrencyPair, regressors: List[str] = None) -> None:
        super().__init__()
        self.currency, self.to_currency = pair
        self.regressors = regressors or []
        self.init_model()

    def init_model(self):
        self.model = Prophet(yearly_seasonality=False, daily_seasonality=False)
        if self.regressors:
            map(self.model.add_regressor, self.regressors)

    def format_data(self, data: pd.DataFrame, y_column: str) -> pd.DataFrame:
        data = pd.DataFrame({
            'ds': data.index,
            'y': np.log(data[y_column]),
            **{regressor: data[regressor] for regressor in self.regressors}
        }).reset_index(drop=True)

        return data

    def learn(self, data: pd.DataFrame, y_column: str = None) -> None:
        logger.debug('Learning from data...')

        if 'ds' not in data.keys():
            data = self.format_data(data, y_column)

        start = time.time()
        self.model.fit(data)

        logger.debug(f'Training took {round(time.time() - start, 4)} seconds.')

    def predict(self, periods: int) -> pd.DataFrame:
        logger.debug('Forecasting...')

        last_index = self.model.history.last_valid_index()
        frequency = self.model.history.ds[last_index] - self.model.history.ds[last_index - 1]

        future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = self.model.predict(future)

        return forecast

    def plot(
        self, forecast: pd.DataFrame, test_data: pd.DataFrame = None, real_scale: bool = True, show: bool = False
    ) -> matplotlib.figure.Figure:
        logger.debug('Plotting...')

        history = self.model.history
        if real_scale:
            forecast = forecast.assign(
                yhat=lambda d: np.exp(d.yhat),
                yhat_lower=lambda d: np.exp(d.yhat_lower),
                yhat_upper=lambda d: np.exp(d.yhat_upper),
            )
            history = self.model.history.assign(y=lambda d: np.exp(d.y))
            if test_data is not None:
                test_data = test_data.assign(y=lambda d: np.exp(d.y))

        forecast = forecast.set_index('ds')
        history = history.set_index('ds')
        test_data = test_data.set_index('ds')

        highest_datapoint = max(history.y.max(), forecast.yhat.max())
        lowest_datapoint = min(history.y.min(), forecast.yhat.min())
        if test_data is not None:
            highest_datapoint = max(highest_datapoint, test_data.y.max())
            lowest_datapoint = min(lowest_datapoint, test_data.y.min())

        fig, ax1 = plt.subplots()
        ax1.set_ylim(bottom=lowest_datapoint - (lowest_datapoint / 2), top=highest_datapoint + (highest_datapoint / 2))

        ax1.plot(history.y, color='red', linewidth=config.plot.linewidth)
        if test_data is not None:
            ax1.plot(test_data.y, color='orange', linewidth=config.plot.linewidth)

        ax1.plot(forecast.yhat, color='black', linestyle=':', linewidth=config.plot.linewidth + 0.2)
        ax1.fill_between(forecast.index, forecast.yhat_upper, forecast.yhat_lower, alpha=0.5, color='darkgray')
        ax1.set_title(
            f'{self.currency}/{self.to_currency} Price (Orange) vs {self.currency}/{self.to_currency} Price Forecast (Black)'
        )
        ax1.set_ylabel(f'{self.currency}/{self.to_currency} Price')
        ax1.set_xlabel('Date')

        legend = ax1.legend()

        if test_data is not None:
            legend.get_texts()[0].set_text('Actual Price (training data)')
            legend.get_texts()[1].set_text('Actual Price (test data)')
            legend.get_texts()[2].set_text('Forecasted Price')
        else:
            legend.get_texts()[0].set_text('Actual Price')
            legend.get_texts()[1].set_text('Forecasted Price')

        if show:
            plt.show()

        return fig

    def test_prediction(
        self, data: pd.DataFrame, y_column: str, training_test_ratio: float, predict_to_datetime: datetime = None
    ) -> None:
        self.init_model()

        timeframe = data.index[-1] - data.index[-2]
        data = self.format_data(data, y_column)
        training_data, test_data = self.training_test_split(data, training_test_ratio)

        self.learn(training_data)

        if predict_to_datetime:
            last_training_datapoint = training_data.ds[training_data.last_valid_index()]
            periods = int((predict_to_datetime - last_training_datapoint).total_seconds() // timeframe.total_seconds())
        else:
            periods = len(test_data)

        forecast = self.predict(periods=periods)

        self.plot(forecast, real_scale=True, test_data=test_data, show=True)
