import datetime as dt
from util import get_data
import numpy as np
import pandas as pd


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cici"


def percent_Bollinger_band(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=20):
    symbols = np.array([symbol])
    # get sufficient data
    Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed))
    if symbol != 'SPY':
        price = Prices_W_SPY[symbols]
    else:
        price = Prices_W_SPY[['SPY']]

    # calculate SMA
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    # calculate BB %
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    result = bbp.loc[sd:, :]

    return result


def relative_Strength_index(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14):
    symbols = np.array([symbol])
    Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed))

    if symbol != 'SPY':
        price = Prices_W_SPY[symbols]
    else:
        price = Prices_W_SPY[['SPY']]

    daily_rets = price.copy()
    daily_rets.values[1:, :] = price.values[1:, :] - price.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = price.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]

    down_loss = price.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]

    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:lookback, :] = np.nan

    rsi[rsi == np.inf] = 100
    result = rsi.loc[sd:, :]

    return result


def momentum(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=12):
    symbols = np.array([symbol])
    Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed))
    price = Prices_W_SPY.loc[:, Prices_W_SPY.columns != 'SPY']

    shifted_price = price.shift(lookback)
    roc = ((price - shifted_price) / shifted_price) * 100
    result = roc.loc[sd:, :]
    return result


def stochastic_Indicator(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14):
    symbols = np.array([symbol])
    adjusted_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed))
    if symbol != 'SPY':
        adj_close_price = adjusted_Prices_W_SPY[symbols]
    else:
        adj_close_price = adjusted_Prices_W_SPY[['SPY']]


    closed_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="Close")
    if symbol != 'SPY':
        close_price = closed_Prices_W_SPY[symbols]
    else:
        close_price = closed_Prices_W_SPY[['SPY']]

    adjust_factor = adj_close_price / close_price

    high_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="High")
    if symbol != 'SPY':
        high_price = high_Prices_W_SPY[symbols]
    else:
        high_price = high_Prices_W_SPY[['SPY']]

    adj_high_price = high_price * adjust_factor
    adj_highest_high = adj_high_price.rolling(window=lookback, min_periods=lookback).max()

    low_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="Low")
    if symbol != 'SPY':
        low_price = low_Prices_W_SPY[symbols]
    else:
        low_price = low_Prices_W_SPY[['SPY']]
    adj_low_price = low_price * adjust_factor
    adj_lowest_low = adj_low_price.rolling(window=lookback, min_periods=lookback).min()

    k = (adj_close_price - adj_lowest_low) / (adj_highest_high - adj_lowest_low) * 100
    result = k.loc[sd:, :]
    return result


def commodity_Channel_index(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=20):
    symbols = np.array([symbol])
    adjusted_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed))
    if symbol != 'SPY':
        adj_close_price = adjusted_Prices_W_SPY[symbols]
    else:
        adj_close_price = adjusted_Prices_W_SPY[['SPY']]

    closed_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="Close")
    if symbol != 'SPY':
        close_price = closed_Prices_W_SPY[symbols]
    else:
        close_price = closed_Prices_W_SPY[['SPY']]

    adjust_factor = adj_close_price / close_price

    high_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="High")
    if symbol != 'SPY':
        high_price = high_Prices_W_SPY[symbols]
    else:
        high_price = high_Prices_W_SPY[['SPY']]
    adj_high_price = high_price * adjust_factor

    low_Prices_W_SPY = get_data(symbols, pd.date_range(sd - dt.timedelta(days=lookback * 2), ed), colname="Low")
    if symbol != 'SPY':
        low_price = low_Prices_W_SPY[symbols]
    else:
        low_price = low_Prices_W_SPY[['SPY']]
    adj_low_price = low_price * adjust_factor

    tp = (adj_high_price + adj_low_price + adj_close_price) / 3
    tp_sma = tp.rolling(window=20, min_periods=20).mean()

    def subprocess(x):
        val_tp = tp.loc[x.index, symbols[0]]
        val_tp_sma = tp_sma.loc[x.index, symbols[0]]
        return abs(val_tp - val_tp_sma[-1]).sum() / 20

    md = tp.rolling(window=lookback).apply(subprocess, raw=False)

    cci = (tp - tp_sma) / (0.015 * md)
    result = cci.loc[sd:, :]
    return result
