"""Utility for stress testing strategies on historical crash periods.

This script underpins шаг 2.5 из roadmap: после стандартных бэктестов
выполните стресс‑тесты на резких обвалах рынка. Укажите тикер и
период — модуль скачает котировки через `yfinance`, прогонит простую
стратегию пересечения SMA в `backtrader` и выведет P&L, максимальную
просадку и список исполненных ордеров.
"""

import backtrader as bt
import yfinance as yf
import pandas as pd


class SmaCross(bt.Strategy):
    params = dict(fast=3, slow=7)

    def __init__(self):
        self.order_log = []
        fast = bt.ind.SMA(period=self.p.fast)
        slow = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(fast, slow)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = 'BUY' if order.isbuy() else 'SELL'
            self.order_log.append(
                {
                    'date': bt.num2date(order.executed.dt).strftime('%Y-%m-%d'),
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'action': action,
                }
            )


def run_crash_scenario(symbol: str, start: str, end: str):
    """Replay a historical crash period and return P&L, drawdown and orders."""
    data = yf.download(symbol, start=start, end=end, group_by="ticker")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, level='Ticker', axis=1)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    datafeed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(datafeed)

    start_cash = 10000
    cerebro.broker.setcash(start_cash)

    # Автоматически подбираем размер позиции, чтобы тесты работали
    # даже для дорогих активов вроде BTC: покупаем долю, на которую
    # хватает стартового капитала.
    max_price = float(data['Close'].max())
    stake = max(start_cash / max_price, 0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    result = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    dd = result.analyzers.dd.get_analysis()['max']['drawdown']
    return {
        'pnl': final_value - start_cash,
        'max_drawdown': dd,
        'orders': result.order_log,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay crash periods and report stats")
    parser.add_argument("symbol", help="Ticker symbol, e.g. BTC-USD")
    parser.add_argument("start", help="Start date in YYYY-MM-DD")
    parser.add_argument("end", help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    res = run_crash_scenario(args.symbol, args.start, args.end)
    print(f"P&L: {res['pnl']:.2f}")
    print(f"Max drawdown: {res['max_drawdown']:.2f}%")
    print('Orders:')
    for o in res['orders']:
        print(o)
