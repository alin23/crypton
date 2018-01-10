# -*- coding: utf-8 -*-

from ccxt.base.errors import ExchangeError
from ccxt.async.base.exchange import Exchange


class CryptoCompare(Exchange):
    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                'id': 'cryptocompare',
                'name': 'Cryptocompare',
                'rateLimit': 6000,
                'countries': 'US',
                'hasCORS': False,
                # obsolete metainfo interface
                'hasFetchTickers': True,
                'hasFetchOrder': False,
                'hasFetchOrders': False,
                'hasFetchOpenOrders': False,
                'hasFetchClosedOrders': False,
                'hasFetchMyTrades': False,
                'hasFetchCurrencies': True,
                'hasDeposit': False,
                'hasWithdraw': False,
                # new metainfo interface
                'has': {
                    'fetchTickers': True,
                    'fetchOrder': False,
                    'fetchOrders': False,
                    'fetchOpenOrders': False,
                    'fetchClosedOrders': False,
                    'fetchMyTrades': False,
                    'fetchCurrencies': True,
                    'deposit': False,
                    'withdraw': False,
                },
                'urls': {
                    'logo':
                        'https://www.cryptocompare.com/media/19990/logo-bold.svg',
                    'api': {
                        'public': 'https://www.cryptocompare.com/api/data',
                        'min': 'https://min-api.cryptocompare.com/data',
                    },
                    'www':
                        'https://www.cryptocompare.com',
                    'doc': [
                        'https://www.cryptocompare.com/api',
                        'https://www.cryptocompare.co.nz/Forum/Thread/255',
                        'https://www.cryptocompare.co.nz/Forum/Thread/256',
                    ],
                },
                'api': {
                    'public': {
                        'get': [
                            'coinlist',
                            'coinsnapshotfullbyid',
                            'coinsnapshot',
                            'socialstats',
                            'miningcontracts',
                            'miningequipment',
                        ],
                    },
                    'min': {
                        'get': [
                            'price',
                            'pricemulti',
                            'pricemultifull',
                            'pricehistorical',
                            'generateAvg',
                            'dayAvg',
                            'subsWatchlist',
                            'subs',
                            'exchanges',
                            'exchanges',
                            'volumes',
                            'pairs',
                            'histoday',
                            'histohour',
                            'histominute',
                        ],
                    },
                },
            }
        )

    def sign(self, path, api='public', method='GET', params={}, headers=None, body=None):
        url = '{}/{}?{}'.format(self.urls['api'][api], path, self.urlencode(params))
        return {'url': url, 'method': method, 'body': body, 'headers': headers}

    async def request(self, path, api='public', method='GET', params={}, headers=None, body=None):
        response = await self.fetch2(path, api, method, params, headers, body)
        if not response:
            raise ExchangeError('Empty response from ' + self.id)

        status = response.get('Response', 'Error')
        if status == 'Error':
            raise ExchangeError(self.id + ' ' + response.get('Message', '') + '\n' + self.json(response))

        return response
