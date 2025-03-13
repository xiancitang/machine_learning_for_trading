import datetime as dt
import numpy as np
import pandas as pd
from util import get_data
import indicators as idc


class ManualStrategy(object):
    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "cici"

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    # Manual Rule-Based trade
    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        symbols = np.array([symbol])
        Prices_W_SPY = get_data(symbols, pd.date_range(sd, ed))
        Prices = Prices_W_SPY[symbols]

        rsi = idc.relative_Strength_index(symbol=symbol, sd=sd, ed=ed, lookback=14)
        sti = idc.stochastic_Indicator(symbol=symbol, sd=sd, ed=ed, lookback=20)
        d = sti.rolling(window=3, min_periods=3).mean()
        cci = idc.commodity_Channel_index(symbol=symbol, sd=sd, ed=ed, lookback=20)

        position = 0
        prev_price = 0
        Prices["Trade"] = 0
        Prices["Position"] = 0

        for index, day in Prices.iterrows():

            short_values = [d.loc[index, symbol] > 80, cci.loc[index, symbol] > 100,
                            rsi.loc[index, symbol] > 70]

            long_values = [d.loc[index, symbol] < 20, cci.loc[index, symbol] < -100,
                           rsi.loc[index, symbol] < 30]

            # Sell as much as possible if indicators support short signal
            if sum(short_values) >= 2:
                shares = -1000 - position
                price = day[symbol] * (1 - self.impact)
                gain = (price - prev_price) * shares * (-1) - self.commission
                if gain > 0:
                    Prices.loc[index, "Trade"] = shares
                    position += Prices.loc[index, "Trade"]
                    # if position restore to 0, reset previous price to 0
                    if position == 0:
                        prev_price = 0
                    else:
                        prev_price = price

            # Buy as much as possible if indicators support long signal
            elif sum(long_values) >= 2:
                shares = 1000 - position
                price = day[symbol] * (1 + self.impact)
                # estimate gain ; if previous position <0 , current price < previous price to make profit ;otherwise current price > prev price to make profit
                gain = abs(prev_price - price) * shares - self.commission
                if gain > 0 :
                    Prices.loc[index, "Trade"] = shares
                    position += Prices.loc[index, "Trade"]
                    if position == 0:
                        prev_price = 0
                    else:
                        prev_price = price

            # do nothing if no signal
            else:
                Prices.loc[index, "Trade"] = 0
                position += Prices.loc[index, "Trade"]
            Prices.loc[index, "Position"] = position

        Trade = Prices["Trade"].copy()
        result = pd.DataFrame({symbol: Trade.values}, index=Trade.index)
        return result


# Benchmark Trade
def testBenchmark(symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    symbol = np.array([symbol])
    Prices_W_SPY = get_data(symbol, pd.date_range(sd, ed))
    Prices = Prices_W_SPY.loc[:, Prices_W_SPY.columns != 'SPY']
    bench_trades = Prices.copy()

    bench_trades["Trade"] = 0
    bench_trades.iloc[0, bench_trades.columns.get_loc("Trade")] = 1000

    Trade = bench_trades["Trade"].copy()
    result = pd.DataFrame({symbol[0]: Trade.values}, index=Trade.index)
    return result