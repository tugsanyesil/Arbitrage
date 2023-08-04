import ccxt

print(ccxt.exchanges)


binance = ccxt.binance()
orderbook = binance.fetch_order_book('BTC/USDT')

print(orderbook)