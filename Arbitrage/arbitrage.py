import itertools

import pandas as pd

from Market.Binance.binancemarket import BinanceMarket
import numpy as np

np.set_printoptions(suppress=True)


def order_2_str(order):
    return "BUY" if order else "SELL"


def array_index_2_matrix_index(index, length):
    return index % length, int(index / length)


class Arbitrage:

    def __init__(self, market):
        self.market = market
        self.assets = []
        self.assets_length = len(self.assets)
        self.price_matrix = []
        # self.qties_matrix = []
        self.validation_matrix = []
        self.validated_symbol_indexes = []
        self.price_indexes = []
        self.validated_price_symbol_indexes = []
        self.base = -1
        self.path_length = 0
        self.paths = []
        self.paths_symbols = []
        self.paths_prices = []
        self.paths_orders = []
        self.paths_fee_factors = []
        self.paths_quantities = []
        self.actual_path_quantities = []
        self.profits = np.zeros(len(self.paths))
        self.max_profit_length = 1
        self.min_profit = 1
        self.max_profit_indexes = []

    def initialize(self, assets, base, path_length, max_profit_length=1, min_profit=1):
        self.max_profit_length = max_profit_length
        self.min_profit = min_profit
        self.set_symbols(assets)
        self.set_base(base)
        self.set_path_length(path_length)
        self.set_matrix()
        self.set_paths()

    def loop(self):
        self.set_matrix()
        self.set_profits()
        self.decide_about_profits()

    def pair_2_symbol(self, pair):
        return self.assets[pair[0]] + self.assets[pair[1]]

    def set_symbols(self, assets):
        if assets == "All":
            self.assets = self.market.assets
        elif assets == "Stabil Coins":
            self.assets = self.market.get_USD_assets()
        else:
            self.assets = assets

        self.assets_length = len(self.assets)
        print("self.assets_length : ", self.assets_length)
        # print("self.assets : ", len(self.assets), self.assets)

        pairs = [(i, j) for j in range(self.assets_length) for i in range(self.assets_length)]
        # print("pairs : ", len(pairs), pairs)

        symbols = [None if pair[0] == pair[1] else self.pair_2_symbol(pair) for pair in pairs]
        if self.assets.__contains__("T") and self.assets.__contains__("TUSD"):  # TODO
            T_index = self.assets.index("T")
            TUSD_index = self.assets.index("TUSD")
            fake_TUSDT_index = self.assets_length * T_index + TUSD_index
            symbols[fake_TUSDT_index] = None
        # print("symbols : ", len(symbols), symbols)

        self.validated_symbol_indexes = self.market.get_validated_symbol_indexes(symbols)
        symbols_validated = [symbols[index] for index in self.validated_symbol_indexes]
        # print("self.validated_symbols_indexes : ", len(self.validated_symbol_indexes), self.validated_symbol_indexes)
        # print("symbols_validated : ", len(symbols_validated), symbols_validated)

        self.market.set_symbols(symbols_validated)

        price_symbols = self.market.get_symbols()
        self.validated_price_symbol_indexes = [index for index, price_symbol in enumerate(price_symbols) if self.market.symbols.__contains__(price_symbol)]
        validated_price_symbols = [price_symbols[index] for index in self.validated_price_symbol_indexes]
        self.price_indexes = [validated_price_symbols.index(symbol) for symbol in symbols_validated]

        self.price_matrix = np.zeros([self.assets_length, self.assets_length], dtype=float)
        # self.qties_matrix = np.zeros([self.assets_length, self.assets_length], dtype=float)
        self.validation_matrix = np.zeros([self.assets_length, self.assets_length], dtype=bool)
        [self.validation_matrix.itemset(validated_symbol_index, True) for validated_symbol_index in self.validated_symbol_indexes]

        print(len(self.validated_price_symbol_indexes), self.validated_price_symbol_indexes)
        print(len(validated_price_symbols), validated_price_symbols)
        # print("self.validation_matrix\n", self.validation_matrix)
        # print()

    def set_matrix(self):
        self.price_matrix = np.zeros([self.assets_length, self.assets_length], dtype=float)
        # self.qties_matrix = np.zeros([self.assets_length, self.assets_length], dtype=float)
        # matrix = np.zeros([self.assets_length, self.assets_length], dtype=float)
        prices, qties = self.market.get_prices()
        prices = [prices[i] for i in self.validated_price_symbol_indexes]
        qties = [qties[i] for i in self.validated_price_symbol_indexes]
        # print(prices)
        for index, symbol_index in enumerate(self.validated_symbol_indexes):
            self.price_matrix.itemset(symbol_index, prices[self.price_indexes[index]][1])
            # self.qties_matrix.itemset(symbol_index, qties[self.price_indexes[index]][1])

        transpoze_price_matrix = self.price_matrix.transpose().copy()
        transpoze_price_matrix[transpoze_price_matrix == 0] = np.inf
        transpoze_price_matrix = 1 / transpoze_price_matrix

        for index, symbol_index in enumerate(self.validated_symbol_indexes):
            self.price_matrix.itemset(symbol_index, prices[self.price_indexes[index]][0])
            # self.qties_matrix.itemset(symbol_index, qties[self.price_indexes[index]][0])

        self.price_matrix = self.price_matrix + transpoze_price_matrix

        # df = pd.DataFrame(self.price_matrix)
        # df.columns = self.assets
        # df.index = self.assets
        # print("matrix\n", df)
        # print()

    def set_base(self, base):
        self.base = base if isinstance(base, int) else self.assets.index(base)

    def set_path_length(self, path_length):
        self.path_length = path_length

        self.paths_symbols = [None] * self.max_profit_length
        self.paths_prices = [None] * self.max_profit_length
        self.paths_orders = [None] * self.max_profit_length
        self.paths_fee_factors = [None] * self.max_profit_length
        self.paths_quantities = [None] * self.max_profit_length
        self.actual_path_quantities = [None] * (self.path_length + 1)

        for i in range(self.max_profit_length):
            self.paths_symbols[i] = [None] * self.path_length
            self.paths_prices[i] = [None] * self.path_length
            self.paths_orders[i] = [None] * self.path_length
            self.paths_fee_factors[i] = [None] * self.path_length
            self.paths_quantities[i] = [None] * (self.path_length + 1)

    def set_paths(self):
        neighbor_list = list(range(self.assets_length))

        delete_list = []

        for index, neighbor in enumerate(neighbor_list):
            if 0 == self.price_matrix[neighbor, self.base]:
                delete_list.append(index)

        delete_list.reverse()

        for index in delete_list:
            neighbor_list.pop(index)

        *paths, = itertools.permutations(neighbor_list, self.path_length - 1)
        # print(paths)
        print("paths_len : ", len(paths))
        self.paths = []
        for path in paths:
            delete = False
            for j in range(self.path_length - 2):
                if 0 == self.price_matrix[path[j + 1], path[j]]:
                    delete = True
                    break
            if not delete:
                path = list(path)
                path.insert(0, self.base)
                path.append(self.base)
                self.paths.append(path)

        print("paths_len : ", len(self.paths))
        # print()

    def set_profits(self):
        self.profits = np.ones(len(self.paths))
        for i, path in enumerate(self.paths):
            for j in range(self.path_length):
                self.profits[i] *= self.price_matrix[path[j + 1], path[j]]

        # print("profits\n", pd.DataFrame(self.profits))

    def execute_path(self, profit_index):
        print("execute_path")
        path_index = self.max_profit_indexes[profit_index]
        self.actual_path_quantities[0] = self.paths_quantities[profit_index][0]

        for order_index in range(self.path_length):
            print("order_index : ", order_index, "self.actual_path_quantities : ", self.actual_path_quantities)
            self.market.market_order_safely(self.paths_symbols[profit_index][order_index], self.paths_orders[profit_index][order_index], self.actual_path_quantities[order_index])
            # self.market.last_trade_check(self.paths_symbols[path_index][order_index])
            self.market.refresh_account()
            self.actual_path_quantities[order_index + 1] = self.market.free_quantity(self.assets[self.paths[path_index][order_index + 1]])
        print("self.actual_path_quantities : ", self.actual_path_quantities)

    def decide_about_profits(self):

        self.max_profit_indexes = self.profits.argsort()[-self.max_profit_length:][::-1]
        for profit_index, path_index in enumerate(self.max_profit_indexes):

            path = self.paths[path_index]
            for order_index in range(self.path_length):  # set_final_profits
                pair = [path[order_index], path[order_index + 1]]
                self.paths_fee_factors[profit_index][order_index] = self.market.get_fee_factor(self.assets[pair[0]], self.assets[pair[1]])
                self.profits[path_index] *= self.paths_fee_factors[profit_index][order_index]

            print("%d.  max_profit_index : %d  max_profit : %f" % (profit_index, path_index, self.profits[path_index]))
            if self.profits[path_index] > self.min_profit:

                self.paths_quantities[profit_index][0] = self.market.free_quantity(self.assets[self.base])
                for order_index in range(self.path_length):
                    pair = [path[order_index], path[order_index + 1]]
                    self.paths_quantities[profit_index][order_index + 1] = self.price_matrix[pair[1], pair[0]] * self.paths_quantities[profit_index][order_index]
                    if not self.market.wallet_includes_bnb:
                        self.paths_quantities[profit_index][order_index + 1] *= self.paths_fee_factors[profit_index][order_index]

                    self.paths_orders[profit_index][order_index] = self.validation_matrix[pair[0], pair[1]]
                    if self.paths_orders[profit_index][order_index]:
                        pair.reverse()
                    self.paths_orders[profit_index][order_index] = order_2_str(self.paths_orders[profit_index][order_index])
                    self.paths_prices[profit_index][order_index] = self.price_matrix[pair[1], pair[0]]
                    self.paths_symbols[profit_index][order_index] = self.pair_2_symbol(pair)

                path_fees = [100 * (1 - fee_factor) for fee_factor in self.paths_fee_factors[profit_index]]
                print("path : ", path)
                print("path_symbols : ", self.paths_symbols[profit_index])
                print("path_prices : ", self.paths_prices[profit_index])
                print("path_orders : ", self.paths_orders[profit_index])
                print("path_fees : ", ["%.3f%%" % fee for fee in path_fees])
                print("path_quantities : ", self.paths_quantities[profit_index])

                if self.market.client and profit_index == 0:
                    self.execute_path(profit_index)
                print()


if __name__ == '__main__':
    import time

    my_market = BinanceMarket()
    # my_market = BinanceMarket(client=True)

    arbitrage = Arbitrage(my_market)

    my_assets = "All"
    # my_assets = "Stabil Coins"  # my_assets = ["BUSD", "TUSD", "USDC", "USDP", "USDT"]
    # my_assets = ["BTC", "ETH", "BNB", "SOL", "BUSD"]
    # my_assets = ["GMT", "VEN", "BNB", "WIN", "BUSD"]
    initialize_time_start = time.time()
    arbitrage.initialize(assets=my_assets, base="BUSD", path_length=3, max_profit_length=3, min_profit=1)
    initialize_time_end = time.time()
    print("initialize_time : ", initialize_time_end - initialize_time_start, "\n")

    # arbitrage.loop()

    while 1:
        loop_time_start = time.time()
        arbitrage.loop()
        loop_time_end = time.time()
        print("loop_time : ", loop_time_end - loop_time_start)
        print()
        time.sleep(0.2)


