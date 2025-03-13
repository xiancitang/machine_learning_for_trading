import ManualStrategy as ms
import marketsimcode as mi
import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner as sl


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cici"


def experiment1():
    sv = 100000
    symbol = 'JPM'
    commission = 9.95
    impact = 0.005
    in_sp_sd = dt.datetime(2008, 1, 1)
    in_sp_ed = dt.datetime(2009, 12, 31)
    out_sp_sd = dt.datetime(2010, 1, 1)
    out_sp_ed = dt.datetime(2011, 12, 31)

    # Manual Strategy
    ms_learner = ms.ManualStrategy(verbose=False, commission=commission, impact=impact)

    in_ms_trades = ms_learner.testPolicy(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed, sv=sv)
    out_ms_trades = ms_learner.testPolicy(symbol=symbol, sd=out_sp_sd, ed=out_sp_ed, sv=sv)

    in_ms_portval = mi.compute_portvals(in_ms_trades, start_val=sv, commission=commission, impact=impact)
    out_ms_portval = mi.compute_portvals(out_ms_trades, start_val=sv, commission=commission, impact=impact)

    # Benchmark
    in_bench_trades = ms.testBenchmark(symbol=symbol,sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    out_bench_trades = ms.testBenchmark(symbol=symbol, sd=out_sp_sd, ed=out_sp_ed,sv=sv)
    in_bench_portval = mi.compute_portvals(in_bench_trades, start_val=sv, commission=commission, impact=impact)

    out_bench_portval = mi.compute_portvals(out_bench_trades, start_val=sv, commission=commission, impact=impact)

    #  Strategy Q Learner

    q_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    q_learner.add_evidence(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    in_ql_trade = q_learner.testPolicy(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    out_ql_trade = q_learner.testPolicy(symbol=symbol,sd=out_sp_sd, ed=out_sp_ed,sv=sv)
    in_ql_portval = mi.compute_portvals(in_ql_trade, start_val=sv, commission=commission, impact=impact)
    out_ql_portval = mi.compute_portvals(out_ql_trade, start_val=sv, commission=commission, impact=impact)


    # In Sample Chart
    norm_in_ms_portval = in_ms_portval / in_ms_portval.iloc[0]
    norm_in_ql_portval = in_ql_portval / in_ql_portval.iloc[0]
    norm_in_bench_portval = in_bench_portval / in_bench_portval.iloc[0]

    plt.plot(norm_in_ms_portval, '-', label='Manual Strategy')
    plt.plot(norm_in_ql_portval, '-', label='Q Learner')
    plt.plot(norm_in_bench_portval, '-', label='Benchmark')
    plt.xticks(rotation=30, ha='right')
    plt.title("In Sample - Manual Strategy vs QL vs Benchmark")
    plt.ylabel("Normed Portfolio Values")
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid()
    plt.savefig('./images/EXP1_in_Sample.png')
    plt.close()

    # Out of sample Chart
    norm_out_ms_portval = out_ms_portval / out_ms_portval.iloc[0]
    norm_out_ql_portval = out_ql_portval / out_ql_portval.iloc[0]
    norm_out_bench_portval = out_bench_portval / out_bench_portval.iloc[0]

    plt.plot(norm_out_ms_portval, '-', label='Manual Strategy')
    plt.plot(norm_out_ql_portval, '-', label='Q Learner')
    plt.plot(norm_out_bench_portval, '-', label='Benchmark')
    plt.xticks(rotation=30, ha='right')
    plt.title("Out Sample _ Manual Strategy vs QL vs Benchmark")
    plt.ylabel("Normed Portfolio Values")
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid()
    plt.savefig('./images/EXP1_out_Sample.png')
    plt.close()
