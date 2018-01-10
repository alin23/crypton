from typing import Any, Dict, List, Tuple, Coroutine, AsyncIterator
from datetime import datetime, timedelta

import pandas as pd

from .. import CACHE_DIR, api, config, logger
from ..types import TimeFrame, CurrencyPair

CRYPTOCOMPARE = api.CryptoCompare()


class CryptoCompare:
    def __init__(
        self,
        pair: CurrencyPair,
        datapoint_fetch_limit: int = config.cryptocompare.datapoint_fetch_limit,
        timeframe: TimeFrame = TimeFrame.HOUR,
    ) -> None:
        self.datapoint_fetch_limit = datapoint_fetch_limit
        self.timeframe = timeframe
        self.currency, self.to_currency = pair

        self.cache_file_name = CACHE_DIR / f'{self.currency}_{self.to_currency}_{self.timeframe.name}.msg'
        self.data = self.get_cached_dataframe()

    def get_timestamps(self, time_range: Tuple[float, float]) -> List[float]:
        from_datetime, to_datetime = time_range
        timeframes = f'{self.timeframe.name.lower()}s'

        count = int((to_datetime - from_datetime) / self.timeframe / self.datapoint_fetch_limit)
        if count == 0:
            return [to_datetime]

        timestamps = [
            datetime.fromtimestamp(to_datetime) - timedelta(**{
                timeframes: self.datapoint_fetch_limit * i
            }) for i in range(count)
        ]

        last_timestamp = timestamps[-1] - timedelta(**{timeframes: self.datapoint_fetch_limit})
        if last_timestamp.timestamp() - from_datetime > self.timeframe * 2:
            timestamps.append(last_timestamp)

        return [t.timestamp() for t in timestamps]

    def get_cached_dataframe(self) -> pd.DataFrame:
        if not self.cache_file_name.exists():
            return None

        return pd.read_msgpack(self.cache_file_name)

    def cache_dataframe(self) -> None:
        self.data.to_msgpack(self.cache_file_name)

    def fetch_history(self, **params):
        if 'limit' in params:
            params['limit'] = int(params['limit'])
        if 'toTs' in params:
            params['toTs'] = int(params['toTs'])
        return getattr(CRYPTOCOMPARE, f'min_get_histo{self.timeframe.name.lower()}')(params=params)

    def get_requests(self, time_range: Tuple[float, float]) -> List[Coroutine[None, None, Dict[str, Any]]]:
        from_datetime, _ = time_range

        params = {
            'fsym': self.currency,
            'tsym': self.to_currency,
        }
        if from_datetime is None:
            requests = [self.fetch_history(limit=self.datapoint_fetch_limit, **params)]
        else:
            timestamps = self.get_timestamps(time_range)
            last_datapoints = min(self.datapoint_fetch_limit, (timestamps[-1] - from_datetime) / self.timeframe)
            requests = [
                self.fetch_history(limit=self.datapoint_fetch_limit, toTs=ts, **params) for ts in timestamps[:-1]
            ]
            requests.append(self.fetch_history(limit=last_datapoints, toTs=timestamps[-1], **params))

        return requests

    def data_between(self, from_timestamp: float, to_timestamp: float):
        params = dict(microsecond=0, second=0)
        from_datetime = datetime.fromtimestamp(from_timestamp)
        to_datetime = datetime.fromtimestamp(to_timestamp)
        if self.timeframe > TimeFrame.MINUTE:
            params['minute'] = 0
        if self.timeframe > TimeFrame.HOUR:
            params['hour'] = 0

        from_datetime = from_datetime.replace(**params)
        to_datetime = to_datetime.replace(**params)
        return self.data.query('@from_datetime <= index & index <= @to_datetime')

    async def iter_responses(self,
                             requests: List[Coroutine[None, None, Dict[str, Any]]]) -> AsyncIterator[Dict[str, Any]]:
        for request in requests:
            response = await request
            if response is not None:
                if response.get('Response') != 'Success':
                    logger.error(response)
                    continue
                yield response
            logger.warning(f'Response is none for {request}')

    async def get_dataframe(self, requests: List[Coroutine[None, None, Dict[str, Any]]]) -> pd.DataFrame:
        data = None
        async for response in self.iter_responses(requests):
            new_data = (
                pd.DataFrame(response['Data']).assign(time=lambda d: d.time.apply(datetime.fromtimestamp)
                                                      ).set_index('time').sort_index()
            )
            if data is None:
                data = new_data
            else:
                data = data.append(new_data)
        if data is not None:
            return data[~data.index.duplicated()]

    async def get_price_history(self, time_range: Tuple[float, float] = None) -> pd.DataFrame:
        from_datetime, to_datetime = time_range or (None, datetime.now().timestamp())
        data: pd.DataFrame = None

        if self.data is not None and not self.data.empty:
            requests: List[Coroutine[None, None, Dict[str, Any]]] = []
            first_datapoint = self.data.first_valid_index().timestamp()
            last_datapoint = self.data.last_valid_index().timestamp()
            if to_datetime > last_datapoint:
                requests += self.get_requests((last_datapoint, to_datetime))
            if from_datetime and from_datetime < first_datapoint:
                requests += self.get_requests((from_datetime, first_datapoint))
            if requests:
                data = await self.get_dataframe(requests)
                data = self.data.append(data).sort_index()
                data = data[~data.index.duplicated()]
        else:
            requests = self.get_requests((from_datetime, to_datetime))
            data = await self.get_dataframe(requests)

        if data is not None and not data.empty:
            self.data = data
            self.cache_dataframe()

        if not from_datetime:
            return self.data

        return self.data_between(from_datetime, to_datetime)
