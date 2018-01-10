import asyncio
from datetime import datetime

import kick
import pandas as pd
from dateutil import parser

import fire
import ccxt.async as ccxt

from . import APP_NAME, config, logger
from .types import Date, TimeFrame, CurrencyPair
from .predictor import Predictor, KerasPredictor, ProphetPredictor
from .datasource import CryptoCompare

LOOP = asyncio.get_event_loop()
SUPPORTED_PREDICTORS = ['prophet', 'keras']


class Crypton:
    def __init__(
        self, pair: CurrencyPair, timeframe: TimeFrame = TimeFrame.HOUR, predictor: str = 'prophet', **predictor_args
    ) -> None:
        self.currency, self.to_currency = pair
        self.timeframe = TimeFrame.get(timeframe)

        self.cryptocompare = CryptoCompare(pair, timeframe=self.timeframe)
        self.predictor = self.init_predictor(predictor, pair, **predictor_args)

    def init_predictor(self, name: str, pair: CurrencyPair, *args, **kwargs) -> Predictor:
        name = name.lower()
        assert name in SUPPORTED_PREDICTORS, f'{name} predictor is not supported'
        if name == 'prophet':
            return ProphetPredictor(pair, **kwargs)
        if name == 'keras':
            return KerasPredictor(pair, **kwargs)

        return None

    def fetch_data_between(self, from_datetime: Date, to_datetime: Date = None) -> pd.DataFrame:
        now = datetime.now()
        from_datetime = parser.parse(from_datetime) if isinstance(from_datetime, str) else from_datetime
        if not to_datetime:
            to_datetime = now
        else:
            to_datetime = parser.parse(to_datetime) if isinstance(to_datetime, str) else to_datetime

        assert to_datetime <= now, f'`to_datetime` param is in the future: {to_datetime}'

        time_range = (from_datetime.timestamp(), to_datetime.timestamp())
        data = LOOP.run_until_complete(self.cryptocompare.get_price_history(time_range))

        return data

    def forecast(
        self,
        learn_from_date: str,
        predict_to_date: str = None,
        training_test_ratio: float = 0.75,
        epochs: int = None,
        batch_size: int = None
    ) -> None:
        now = datetime.now()
        predict_to_datetime = parser.parse(predict_to_date) if predict_to_date else now

        to_datetime = predict_to_datetime if predict_to_datetime <= now else now
        data = self.fetch_data_between(learn_from_date, to_datetime)

        if isinstance(self.predictor, KerasPredictor):
            self.predictor.test_prediction(data, 'close', training_test_ratio, epochs=epochs, batch_size=batch_size)
        elif isinstance(self.predictor, ProphetPredictor):
            self.predictor.test_prediction(data, 'close', training_test_ratio, predict_to_datetime=predict_to_datetime)

    @staticmethod
    def update_config(name='config'):
        kick.update_config(APP_NAME.lower(), variant=name)


def main():
    try:
        fire.Fire(Crypton)
    except KeyboardInterrupt:
        print('Going down...')


if __name__ == '__main__':
    main()
