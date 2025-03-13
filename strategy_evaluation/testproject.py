import ManualStrategy as ms
import marketsimcode as mi
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import StrategyLearner as sl
import random as rand
import numpy as np
from util import get_data
import experiment1 as ep1
import experiment2 as ep2


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cici"


def compute_daily_returns(df):
    """Compute daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.iloc[0] = 0
    return daily_returns


def manual_strategy(symbol, commission, impact):
    sv = 100000
    in_sp_sd = dt.datetime(2008, 1, 1)
    in_sp_ed = dt.datetime(2009, 12, 31)
    out_sp_sd = dt.datetime(2010, 1, 1)
    out_sp_ed = dt.datetime(2011, 12, 31)

    learner = ms.ManualStrategy(verbose=False, commission=commission, impact=impact)

    """ In sample """

    # Manual strategy
    in_df_trades = learner.testPolicy(symbol=symbol,sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    in_manual_portval = mi.compute_portvals(in_df_trades, start_val=sv, commission=commission, impact=impact)
    in_df_trades['postion'] = in_df_trades.cumsum()
    in_entry_position = in_df_trades.iloc[in_df_trades['postion'].nonzero() and in_df_trades[symbol].nonzero()]

    in_mn_daily_rets = compute_daily_returns(in_manual_portval)
    in_mn_cr = "%0.6f" % ((in_manual_portval[-1] / in_manual_portval[0]) - 1)
    in_mn_std_daily_ret = "%0.6f" % (in_mn_daily_rets.std())
    in_mn_mean = "%0.6f" % (in_mn_daily_rets.mean())


    # Benchmark
    in_bench_trades = ms.testBenchmark(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed, sv=sv)
    in_benchmark_portval = mi.compute_portvals(in_bench_trades, start_val=sv, commission=commission, impact=impact)
    in_ben_daily_rets = compute_daily_returns(in_benchmark_portval)
    in_ben_cr = "%0.6f" % ((in_benchmark_portval[-1] / in_benchmark_portval[0]) - 1)
    in_ben_std_daily_ret = "%0.6f" % (in_ben_daily_rets.std())
    in_ben_mean = "%0.6f" % (in_ben_daily_rets.mean())

    # In sample Chart
    norm_in_manual_portval = in_manual_portval / in_manual_portval.iloc[0]
    norm_in_bench_portval = in_benchmark_portval / in_benchmark_portval.iloc[0]

    plt.plot(norm_in_manual_portval, '-', color='red', label='Manual Strategy')
    plt.plot(norm_in_bench_portval, '-', color='purple', label='Benchmark')

    for index, amount in in_entry_position .iterrows():
        if amount[0] > 0:
            plt.axvline(x=index, color='blue', linestyle='-')
        elif amount[0] < 0:
            plt.axvline(x=index, color='black', linestyle='-')
    plt.xticks(rotation=30, ha='right')
    plt.title("In Sample - Manual Strategy vs Benchmark")
    plt.ylabel("Normed Portfolio Values")
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid()
    plt.savefig('./images/Manual_Strategy_in_Sample.png')
    plt.close()

    """ Out of  Sample """

    # Manual strategy
    out_df_trades = learner.testPolicy(symbol=symbol, sd=out_sp_sd, ed=out_sp_ed,sv=sv)
    out_manual_portval = mi.compute_portvals(out_df_trades, start_val=sv, commission=commission, impact=impact)
    out_df_trades['postion'] = out_df_trades.cumsum()
    out_entry_position = out_df_trades.iloc[out_df_trades['postion'].nonzero() and out_df_trades[symbol].nonzero()]
    out_mn_daily_rets = compute_daily_returns(out_manual_portval)
    out_mn_cr = "%0.6f" % ((out_manual_portval[-1] / out_manual_portval[0]) - 1)
    out_mn_std_daily_ret = "%0.6f" % (out_mn_daily_rets.std())
    out_mn_mean = "%0.6f" % (out_mn_daily_rets.mean())

    # Benchmark
    out_bench_trades = ms.testBenchmark(symbol=symbol,  sd=out_sp_sd, ed=out_sp_ed, sv=sv)
    out_benchmark_portval = mi.compute_portvals(out_bench_trades, start_val=sv, commission=commission, impact=impact)

    out_ben_daily_rets = compute_daily_returns(out_benchmark_portval)
    out_ben_cr = "%0.6f" % ((out_benchmark_portval[-1] / out_benchmark_portval[0]) - 1)
    out_ben_std_daily_ret = "%0.6f" % (out_ben_daily_rets.std())
    out_ben_mean = "%0.6f" % (out_ben_daily_rets.mean())

    # Out sample Chart
    norm_out_manual_portval = out_manual_portval / out_manual_portval.iloc[0]
    norm_out_bench_portval = out_benchmark_portval / out_benchmark_portval.iloc[0]
    plt.plot(norm_out_manual_portval, '-', color='red', label='Manual Strategy')
    plt.plot(norm_out_bench_portval, '-', color='purple', label='Benchmark')

    for index, amount in out_entry_position.iterrows():
        if amount[0] > 0:
            plt.axvline(x=index, color='blue', linestyle='-')
        elif amount[0] < 0:
            plt.axvline(x=index, color='black', linestyle='-')
    plt.xticks(rotation=30, ha='right')
    plt.title("Out Sample - Manual Strategy vs Benchmark")
    plt.ylabel("Normed Portfolio Values")
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid()
    plt.savefig('./images/Manual_Strategy_out_Sample.png')
    plt.close()

    # Table
    table = [[in_ben_cr, in_ben_std_daily_ret, in_ben_mean], [in_mn_cr, in_mn_std_daily_ret, in_mn_mean],
             [out_ben_cr, out_ben_std_daily_ret, out_ben_mean], [out_mn_cr, out_mn_std_daily_ret, out_mn_mean]]
    df = pd.DataFrame(table, columns=['cumulative return ', 'standard deviation', 'mean'],
                      index=['Benchmark in sample', 'Manual Strategy in sample', 'Benchmark out sample',
                             'Manual Strategy out sample'])
    df.to_csv(r'./images/Manual_Strategy_Table.csv', sep='\t')

if __name__ == "__main__":
    manual_strategy("JPM", 9.95, 0.005)
    rand.seed(903756595)
    ep1.experiment1()
    ep2.experiment2()

