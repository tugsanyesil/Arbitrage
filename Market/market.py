Markets = ["Binance", "KuCoin"]


class Market:
    def __init__(self):
        self.assets = []
        self.symbols = []
        self.base_assets = []
        self.quote_assets = []

    def get_fee_factor(self, asset_1, asset_2):
        pass

    def set_symbols(self, symbols):
        pass

    def get_prices(self):
        pass

    def get_validated_symbol_indexes(self, symbols):
        return [index for index, symbol in enumerate(symbols) if (self.symbols.__contains__(symbol) if symbol else False)]
