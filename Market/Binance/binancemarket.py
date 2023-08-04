import os

import numpy
import numpy as np
import pandas as pd
import requests
import binance
from Market.market import Market
from binance.spot import Spot
from binance.error import Error
import math
from datetime import datetime


def round_down(number, modulo, precision):
    p10 = math.pow(10, precision)
    return modulo * math.floor(number * p10 / modulo) / p10


def get_site(end_point):
    return "https://api.binance.com/api/v3/" + end_point


def get_symbols_string(symbols):
    return str(symbols).replace("'", "\"").replace(" ", "")


# def get_book_price(symbol):
#     json = requests.get(get_site("ticker/bookTicker"), params=dict(symbol=symbol)).json()
#     return
#
#
# def get_all_book_symbols():
#     return [symbol["symbol"] for symbol in requests.get(get_site("ticker/bookTicker")).json()]

def get_order_book_all(symbol, limit=None):
    if limit is None:
        json = requests.get(get_site("depth"), params=dict(symbol=symbol)).json()
    else:
        json = requests.get(get_site("depth"), params=dict(limit=limit, symbol=symbol)).json()
    # print(json)
    bids = json['bids']
    asks = json['asks']
    return [[float(row[0]) for row in bids], [float(row[0]) for row in asks]], [[float(row[1]) for row in bids], [float(row[1]) for row in asks]]


def get_order_book(symbol):
    json = requests.get(get_site("depth"), params=dict(limit=1, symbol=symbol)).json()
    bids = json['bids'][0]
    asks = json['asks'][0]
    return [float(bids[0]), float(asks[0])], [float(bids[1]), float(asks[1])]


def get_book_price(symbol):
    json = requests.get(get_site("ticker/bookTicker"), params=dict(symbol=symbol)).json()
    # print(json)
    return [float(json['bidPrice']), float(json['askPrice'])], [float(json['bidQty']), float(json['askQty'])]


def get_book_symbols(symbols):
    json = requests.get(get_site("ticker/bookTicker"), params=dict(symbols=symbols)).json()
    return [symbol["symbol"] for symbol in json]


def get_book_prices(symbols):
    json = requests.get(get_site("ticker/bookTicker"), params=dict(symbols=symbols)).json()
    return [[float(symbol['bidPrice']), float(symbol['askPrice'])] for symbol in json], [[float(symbol['bidQty']), float(symbol['askQty'])] for symbol in json]


def get_all_book_symbols():
    json = requests.get(get_site("ticker/bookTicker")).json()
    return [symbol["symbol"] for symbol in json]


def get_all_book_prices():
    json = requests.get(get_site("ticker/bookTicker")).json()
    return [[float(symbol['bidPrice']), float(symbol['askPrice'])] for symbol in json], [[float(symbol['bidQty']), float(symbol['askQty'])] for symbol in json]


def get_all_book_prices_and_symbols():
    json = requests.get(get_site("ticker/bookTicker")).json()
    return [symbol["symbol"] for symbol in json], [[float(symbol['bidPrice']), float(symbol['askPrice'])] for symbol in json], [[float(symbol['bidQty']), float(symbol['askQty'])] for symbol in json]


def get_price(symbol):
    json = requests.get(get_site("ticker/price"), params=dict(symbol=symbol)).json()
    return json["price"]


def get_symbols(symbols):
    json = requests.get(get_site("ticker/price"), params=dict(symbols=symbols)).json()
    return [symbol["symbol"] for symbol in json]


def get_prices(symbols):
    json = requests.get(get_site("ticker/price"), params=dict(symbols=symbols)).json()
    return [float(symbol["price"]) for symbol in json]


def get_all_symbols():
    json = requests.get(get_site("ticker/price")).json()
    return [symbol["symbol"] for symbol in json]


def get_all_prices():
    json = requests.get(get_site("ticker/price")).json()
    return [float(symbol["price"]) for symbol in json]


def get_all_prices_and_symbols():
    json = requests.get(get_site("ticker/price")).json()
    return [symbol["symbol"] for symbol in json], [float(symbol["price"]) for symbol in json]


api_key_filename = "../Market/Binance/Binance_API_Key.txt"
secret_key_filename = "../Market/Binance/Binance_Secret_Key.txt"


# api_key_filename = "Binance_API_Key.txt"
# secret_key_filename = "Binance_Secret_Key.txt"


def get_indexes(list, item):
    return [i for i, x in enumerate(list) if x == item]


def pair_2_symbol(pair, is_reverse=False):
    if is_reverse:
        pair.reverse()
    return pair[0] + pair[1]


class BinanceMarket(Market):
    normal_fee = 0.001
    bnb_fee = 0.00075
    busd_fee = 0
    symbol_request_limit = 100

    def get_rid_of_not_in_a_loop(self):
        symbol_indexes = []
        asset_indexes = []
        for index, asset in enumerate(self.assets):
            base_indexes = get_indexes(self.base_assets, asset)
            quote_indexes = get_indexes(self.quote_assets, asset)

            if (len(base_indexes) + len(quote_indexes)) == 1:
                symbol_index = base_indexes + quote_indexes
                symbol_indexes.append(symbol_index[0])
                asset_indexes.append(index)

        print(asset_indexes)
        print(len(self.symbols), self.symbols)
        print(len(self.base_assets), self.base_assets)
        print(len(self.quote_assets), self.quote_assets)
        symbol_indexes.sort(reverse=True)
        print(symbol_indexes, len(symbol_indexes), min(symbol_indexes), max(symbol_indexes))

        for index in symbol_indexes:
            self.symbols.pop(index)
            self.base_assets.pop(index)
            self.quote_assets.pop(index)

        # print("deleted assets : ", [self.assets[index] for index in asset_indexes])

        asset_indexes.reverse()
        for index in asset_indexes:
            self.assets.pop(index)

    def get_is_there_and_is_reverse_condition(self, asset_1, asset_2):
        binary_symbols = [asset_1 + asset_2, asset_2 + asset_1]
        conditions = [self.symbols.__contains__(symbol) for symbol in binary_symbols]
        is_there = sum(conditions) > 0
        return is_there, conditions.index(True) if is_there else False

    # def get_rid_of_small_assets(self):
    #
    #     USD_assets = self.get_USD_assets()
    #
    #     asset_not_connected_with_USD_and_BTC = []
    #     asset_not_connected_with_USD_but_BTC = []
    #
    #     symbol_infos = []
    #     for asset_index, asset in enumerate(self.assets):
    #         is_there = False
    #         for USD_asset_index, USD_asset in enumerate(USD_assets):
    #             if USD_assets.__contains__(asset):
    #                 is_there = True  # symbols_to_check eklenmiyor
    #                 break
    #             else:
    #                 is_there, is_reverse = self.get_is_there_and_is_reverse_condition(asset, USD_asset)
    #                 if is_there:
    #                     is_reverse = is_reverse
    #                     symbol_infos.append((bool(is_reverse), asset_index, USD_asset_index))
    #                     break
    #
    #         if not is_there:
    #             is_there, is_reverse = self.get_is_there_and_is_reverse_condition(asset, "BTC")
    #             if is_there:
    #                 asset_not_connected_with_USD_but_BTC.append((False, asset_index, -1))
    #             else:
    #                 asset_not_connected_with_USD_and_BTC.append(asset_index)
    #
    #     symbol_infos += asset_not_connected_with_USD_but_BTC
    #     symbol_infos = [(is_reverse, asset_index, pair_2_symbol([self.assets[asset_index], "BTC" if USD_BTC_asset_index == -1 else USD_assets[USD_BTC_asset_index]], is_reverse)) for (is_reverse, asset_index, USD_BTC_asset_index) in symbol_infos]
    #
    #     price_symbols, prices = get_all_prices_and_symbols()
    #
    #     symbol_indexes = [price_symbols.index(symbol_info[2]) for symbol_info in symbol_infos]
    #     prices = [prices[symbol_index] for symbol_index in symbol_indexes]
    #
    #     price_of_BTCBUSD = price_symbols.index("BTCBUSD")
    #     for i in range(-len(asset_not_connected_with_USD_but_BTC), 0):
    #         prices[i] *= price_of_BTCBUSD
    #
    #     for i in range(len(symbol_infos)):
    #         if symbol_infos[i][0]:
    #             prices[i] = 1 / prices[i]
    #
    #     price_treshold = 0.05  # [BUSD]
    #     price_tresholded_indexes = [index for index, price in enumerate(prices) if price < price_treshold]
    #
    #     assets_to_delete = [symbol_infos[price_tresholded_index][1] for price_tresholded_index in price_tresholded_indexes]
    #     assets_to_delete += [symbol_info[1] for symbol_info in asset_not_connected_with_USD_and_BTC]
    #     assets_to_delete.sort(reverse=True)
    #     for asset_index in assets_to_delete:
    #         self.assets.pop(asset_index)

    def refresh_exchange_info(self):
        print(get_site("exchangeInfo"))
        json = requests.get(get_site("exchangeInfo")).json()
        symbols = json['symbols']

        for index, symbol in enumerate(symbols):
            if symbol['isSpotTradingAllowed'] and symbol['status'] == "TRADING":
                self.symbols.append(symbol['symbol'])  # todo delete smaller than 5 cent
                self.base_assets.append(symbol['baseAsset'])
                self.quote_assets.append(symbol['quoteAsset'])
                self.filters_list.append(symbol['filters'])

        self.assets = self.base_assets + self.quote_assets
        self.assets = list(np.unique(np.array(self.assets)))

    def __init__(self, client=False):
        super(BinanceMarket, self).__init__()
        self.account = None
        self.balances = None
        self.filters_list = []

        self.refresh_exchange_info()
        print("self.assets_length : ", len(self.assets))

        self.get_rid_of_not_in_a_loop()
        print("get_rid_of_not_in_a_loop")
        print("self.assets_length : ", len(self.assets))

        # self.get_rid_of_small_assets()
        # print("get_rid_of_small_assets")
        # print("self.assets_length : ", len(self.assets))

        self.symbols_string = ""

        if client:

            # print(client.time()) # Get server timestamp
            # print(client.klines("BTCUSDT", "1m"))  # Get klines of BTCUSDT at 1m interval
            # print(client.klines("BNBUSDT", "1h", limit=10)) # Get last 10 klines of BNBUSDT at 1h interval

            with open(api_key_filename, "r") as api_key_file:
                self.API_KEY = api_key_file.read()

            with open(secret_key_filename, "r") as secret_key_file:
                self.SECRET_KEY = secret_key_file.read()

            self.client = Spot(key=self.API_KEY, secret=self.SECRET_KEY)
            self.refresh_account()

            self.wallet_includes_bnb = self.free_quantity("BNB") > 0
        else:
            self.wallet_includes_bnb = True
            self.client = None

    def free_quantity(self, asset):
        if self.balances is None:
            return 100
        else:
            quantity = self.balances.get(asset)
            return quantity['free'] if quantity else 0

    def locked_quantity(self, asset):
        quantity = self.balances.get(asset)
        return quantity['locked'] if quantity else 0

    def quantity(self, asset):
        quantity = self.balances.get(asset)
        return (quantity['locked'] + quantity['free']) if quantity else 0

    def refresh_account(self):
        self.account = self.client.account()
        balances = self.account['balances']  # balances[0] = {'asset': 'BTC', 'free': '0.00000000', 'locked': '0.00000000'}

        self.balances = {}
        for i in range(len(balances)):
            free = float(balances[i]['free'])
            locked = float(balances[i]['locked'])
            if 0 != (free + locked):
                self.balances.update({balances[i]['asset']: {'free': free, 'locked': locked}})

    def set_symbols(self, symbols):
        if len(symbols) > self.symbol_request_limit:
            self.symbols_string = ""
        else:
            self.symbols_string = get_symbols_string(symbols)

    def get_symbols(self):
        if self.symbols_string == "":
            return get_all_book_symbols()
            # return get_all_symbols()
        else:
            return get_book_symbols(self.symbols_string)
            # return get_symbols(self.symbols_string)

    def get_prices(self):
        if self.symbols_string == "":
            return get_all_book_prices()
            # return get_all_prices()
        else:
            return get_book_prices(self.symbols_string)
            # return get_prices(self.symbols_string)

    def get_validated_symbol_indexes(self, symbols):
        return super(BinanceMarket, self).get_validated_symbol_indexes(symbols)

    def get_fee_factor(self, asset_1, asset_2):
        if asset_1 == "BUSD" or asset_2 == "BUSD":
            return 1 - self.busd_fee
        else:
            if self.wallet_includes_bnb:
                return 1 - self.bnb_fee
            else:
                return 1 - self.normal_fee

    # [
    #     {'filterType': 'PRICE_FILTER', 'minPrice': '0.00000100', 'maxPrice': '922327.00000000', 'tickSize': '0.00000100'},
    #     {'filterType': 'PERCENT_PRICE', 'multiplierUp': '5', 'multiplierDown': '0.2', 'avgPriceMins': 5},
    #     {'filterType': 'LOT_SIZE', 'minQty': '0.00010000', 'maxQty': '100000.00000000', 'stepSize': '0.00010000'},
    #     {'filterType': 'MIN_NOTIONAL', 'minNotional': '0.00010000', 'applyToMarket': True, 'avgPriceMins': 5},
    #     {'filterType': 'ICEBERG_PARTS', 'limit': 10},
    #     {'filterType': 'MARKET_LOT_SIZE', 'minQty': '0.00000000', 'maxQty': '1184.49061020', 'stepSize': '0.00000000'},
    #     {'filterType': 'TRAILING_DELTA', 'minTrailingAboveDelta': 10, 'maxTrailingAboveDelta': 2000, 'minTrailingBelowDelta': 10, 'maxTrailingBelowDelta': 2000},
    #     {'filterType': 'MAX_NUM_ORDERS', 'maxNumOrders': 200}, {'filterType': 'MAX_NUM_ALGO_ORDERS', 'maxNumAlgoOrders': 5}
    # ]
    def filter_check(self, symbol, quantity):

        min_notional_filter = next(item for item in self.filters_list[self.symbols.index(symbol)] if item['filterType'] == 'MIN_NOTIONAL')
        lot_size_filter = next(item for item in self.filters_list[self.symbols.index(symbol)] if item['filterType'] == 'LOT_SIZE')

        min_notional_precision, min_notional_modulo = next((i - 1, int(x)) for i, x in enumerate(str(min_notional_filter['minNotional'])) if (x != '0' and x != '.'))
        lot_size_precision, lot_size_modulo = next((i - 1, int(x)) for i, x in enumerate(str(lot_size_filter['stepSize'])) if (x != '0' and x != '.'))

        min_notional_filter['minNotional'] = float(min_notional_filter['minNotional'])
        lot_size_filter['stepSize'] = float(lot_size_filter['stepSize'])

        # min_notional_remaining = quantity % min_notional_filter['minNotional']
        # lot_size_remaining = quantity % lot_size_filter['stepSize']

        if min_notional_precision == -1:
            min_notional_passing = math.floor(quantity / min_notional_modulo)
        else:
            min_notional_passing = round_down(quantity, min_notional_modulo, min_notional_precision)

        if lot_size_precision == -1:
            lot_size_passing = math.floor(quantity / lot_size_modulo)
        else:
            lot_size_passing = round_down(quantity, lot_size_modulo, lot_size_precision)

        print("min_notional_filter : ", min_notional_filter)
        print("lot_size_filter : ", lot_size_filter)

        lot_size_filter['maxQty'] = float(lot_size_filter['maxQty'])
        lot_size_filter['minQty'] = float(lot_size_filter['minQty'])

        if lot_size_passing >= lot_size_filter['minQty']:
            if lot_size_passing <= lot_size_filter['maxQty']:
                check = True
                condition = 'passed'
            else:
                check = False
                condition = 'maxQty'
        else:
            check = False
            condition = 'minQty'

        if not check:
            print('LOT_SIZE', condition)
            lot_size_passing = min_notional_passing

        return check, lot_size_passing

    def market_order_safely(self, symbol, side, quantity):

        check, passing = self.filter_check(symbol, quantity)
        print("check : %s  passing : %f" % (check, passing))
        if check:
            try:
                print("symbol : %s  side : %s  passing : %f" % (symbol, side, passing))
                if side == "BUY":
                    print("BUY")
                    response = self.client.new_order(symbol=symbol, side=side, type='MARKET', quoteOrderQty=passing)
                else:
                    print("SELL")
                    response = self.client.new_order(symbol=symbol, side=side, type='MARKET', quantity=passing)
            except Error as error:
                print("error : ", error)

            print("response : ", response)

    def last_trade_check(self, symbol):

        trades = self.client.my_trades(symbol=symbol)
        trade = trades[-1]
        timestamp = trade['time'] * 0.001
        trade = {'price': trade['price'], 'qty': trade['qty'], 'quoteQty': trade['quoteQty'], 'commission': trade['commission'], 'commissionAsset': trade['commissionAsset'], 'time': trade['time']}
        print(datetime.fromtimestamp(timestamp), trade)

    def get_USD_assets(self):
        return [asset for asset in self.assets if "USD" in asset]


if __name__ == '__main__':
    import time

    # market = BinanceMarket(client=True)
    # market = Binance()

    # while True:
    #     print()
    #     print("book_price : ", get_book_price("ATOMEUR"))
    #     print("order_book : ", get_order_book("ATOMEUR"))

    # print(get_book_symbols(get_symbols_string(["BTCUSDT","ETHUSDT"])))
    # print(get_book_prices(get_symbols_string(["BTCUSDT","ETHUSDT"])))
    # print(get_all_book_symbols())
    # print(get_all_book_prices())
    # print(get_all_book_prices_and_symbols())

    print("book_price : ", get_book_price("BNBUSDT"))
    # while True:
    #     print()
    #     print("book_price : ", get_book_price("BNBUSDT"))
    #     print("order_book : ", get_order_book_all("BNBUSDT"))

    from datetime import datetime
    from matplotlib import pyplot
    from matplotlib.animation import FuncAnimation
    from random import randrange
    import ccxt

    figure = pyplot.figure()

    ccxt_bid_line, = pyplot.plot([], [], linewidth=2)
    ccxt_ask_line, = pyplot.plot([], [], linewidth=2)

    bid_line, = pyplot.plot([], [], linewidth=2)
    ask_line, = pyplot.plot([], [], linewidth=2)

    best_bid_point, = pyplot.plot([], [], "go")
    best_ask_point, = pyplot.plot([], [], "ro")

    pyplot.legend(['ccxt bids', 'ccxt asks', 'API asks', 'API asks', 'API best bid', 'API best ask'])
    pyplot.xlabel('Price')
    pyplot.ylabel('Quantity')

    binance = ccxt.binance()

    limit = 20


    def update(frame):
        prices, qties = get_order_book_all("BNBUSDT", limit=limit)
        best_prices, best_qties = get_book_price("BNBUSDT")
        orderbook = binance.fetch_order_book('BNB/USDT')
        ccxt_prices, ccxt_qties = [[float(row[0]) for row in orderbook['bids']], [float(row[0]) for row in orderbook['asks']]], [[float(row[1]) for row in orderbook['bids']], [float(row[1]) for row in orderbook['asks']]]

        ccxt_bid_prices = ccxt_prices[0][:limit]
        ccxt_ask_prices = ccxt_prices[1][:limit]
        ccxt_bid_qties = ccxt_qties[0][:limit]
        ccxt_ask_qties = ccxt_qties[1][:limit]

        bid_prices = prices[0]
        ask_prices = prices[1]
        bid_qties = qties[0]
        ask_qties = qties[1]

        best_bid_prices = best_prices[0]
        best_ask_prices = best_prices[1]
        best_bid_qties = best_qties[0]
        best_ask_qties = best_qties[1]

        print("ccxt_bid_prices :", ccxt_bid_prices)
        print("     bid_prices :", bid_prices)

        print("ccxt_ask_prices :", ccxt_ask_prices)
        print("     ask_prices :", ask_prices)

        print("ccxt_bid_qties :", ccxt_bid_qties)
        print("     bid_qties :", bid_qties)

        print("ccxt_ask_qties :", ccxt_ask_qties)
        print("     ask_qties :", ask_qties)

        print("best_bid_prices :", best_bid_prices)
        print("best_ask_prices :", best_ask_prices)
        print("best_bid_qties :", best_bid_qties)
        print("best_ask_qties :", best_ask_qties)

        print()

        ccxt_bid_line.set_data(ccxt_bid_prices, ccxt_bid_qties)
        ccxt_ask_line.set_data(ccxt_ask_prices, ccxt_ask_qties)

        bid_line.set_data(bid_prices, bid_qties)
        ask_line.set_data(ask_prices, ask_qties)

        best_bid_point.set_data(best_bid_prices, best_bid_qties)
        best_ask_point.set_data(best_ask_prices, best_ask_qties)

        figure.gca().relim()
        figure.gca().autoscale_view()

        return None,


    animation = FuncAnimation(figure, update, interval=200)

    pyplot.grid()
    pyplot.show()

    # while True:
    #     update(None)
