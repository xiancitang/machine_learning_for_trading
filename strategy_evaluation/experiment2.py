import marketsimcode as mi
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import StrategyLearner as sl
import numpy as np


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

# how impact affect in-sample trading behavior
def experiment2():
    sv = 100000
    symbol = 'JPM'
    commission = 0.00
    in_sp_sd = dt.datetime(2008, 1, 1)
    in_sp_ed = dt.datetime(2009, 12, 31)

    # test 1 :  impact = 0

    impact = 0
    learner1 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner1.add_evidence(symbol=symbol,sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    in_sp_trade1 = learner1.testPolicy(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    portval_no_imp = mi.compute_portvals(in_sp_trade1, start_val=sv, commission=commission, impact=impact)

    daily_rets_no_imp = compute_daily_returns(portval_no_imp)
    cr1 = "%0.6f" % ((portval_no_imp[-1] / portval_no_imp[0]) - 1)
    std_daily_ret1 = "%0.6f" % (daily_rets_no_imp.std())
    mean1 = "%0.6f" % (daily_rets_no_imp.mean())
    sharpe_ratio1 = "%0.6f" % (np.sqrt(252) * np.mean(daily_rets_no_imp - 0.0) / daily_rets_no_imp.std())

    # test 2 :  impact = 0.005

    impact = 0.005
    learner2 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner2.add_evidence(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    in_sp_trade2 = learner2.testPolicy(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    portval_imp2 = mi.compute_portvals(in_sp_trade2, start_val=sv, commission=commission, impact=impact)

    daily_rets_imp2 = compute_daily_returns(portval_imp2)
    cr2 = "%0.6f" % ((portval_imp2[-1] / portval_imp2[0]) - 1)
    std_daily_ret2 = "%0.6f" % (daily_rets_imp2.std())
    mean2 = "%0.6f" % (daily_rets_imp2.mean())
    sharpe_ratio2 = "%0.6f" % (np.sqrt(252) * np.mean(daily_rets_imp2 - 0.0) / daily_rets_imp2.std())

    # test 3  :  impact = 0.01

    impact = 0.01
    learner3 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner3.add_evidence(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    in_sample_trade3 = learner3.testPolicy(symbol=symbol, sd=in_sp_sd, ed=in_sp_ed,sv=sv)
    portval_imp3 = mi.compute_portvals(in_sample_trade3, start_val=sv, commission=commission, impact=impact)

    daily_rets_imp3 = compute_daily_returns(portval_imp3)
    cr3 = "%0.6f" % ((portval_imp3[-1] / portval_imp3[0]) - 1)
    std_daily_ret3 = "%0.6f" % (daily_rets_imp3.std())
    mean3 = "%0.6f" % (daily_rets_imp3.mean())
    sharpe_ratio3 = "%0.6f" % (np.sqrt(252) * np.mean(daily_rets_imp3 - 0.0) / daily_rets_imp3.std())


    plt.plot(daily_rets_no_imp, '-', label='0')
    plt.plot(daily_rets_imp2, '-', label='0.005')
    plt.plot(daily_rets_imp3, '-', label='0.01')

    plt.xticks(rotation=30, ha='right')
    plt.title("impact comparison")
    plt.ylabel("daily return")
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid()
    plt.savefig('./images/EXP2_Chart.png')
    plt.close()


    # In sample table
    in_table = [[cr1, std_daily_ret1, mean1, sharpe_ratio1], [cr2, std_daily_ret2, mean2, sharpe_ratio2],
                [cr3, std_daily_ret3, mean3, sharpe_ratio3]]
    df = pd.DataFrame(in_table, columns=['cumulative return ', 'standard deviation', 'mean', 'sharpe ratio'],
                      index=['impact 0.0', 'impact 0.005', 'impact 0.01'])
    df.to_csv(r'./images/Exp2_Table.csv', sep='\t')
