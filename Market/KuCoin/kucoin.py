from Market.market import Market


class KuCoin(Market):
    def __init__(self):
        super(KuCoin, self).__init__()

    def set_symbols(self, symbols):
        pass

    def get_prices(self):
        return None

    def get_validated_symbol_indexes(self, symbols):
        return super(KuCoin, self).get_validated_symbol_indexes(symbols)