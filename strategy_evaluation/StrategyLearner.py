""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 			  	   		  		 			  		 			 	 	 		 		 			  	   		  		 			  		 			 	 	 		 		 	
"""

import datetime as dt
import time
import numpy as np
import indicators as idc
import pandas as pd
import util as ut
import math
import QLearner as ql


class StrategyLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		  		 			  		 			 	 	 		 		 	
    """

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
        self.prev_price = 0
        self.learner = None
        self.symbol = None
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def discretize(self, indicator, steps):
        """
        convert the indicator to a single integer

        :param indicator: the value of indicator to discretize
        :type indicator: dataframe
        :return: the discretized value of indicator
        :rtype: int
        """
        n = len(indicator)
        stepsize = math.ceil(n / steps)
        indicator = indicator.sort_values(indicator.columns[0])
        threshold = np.zeros((steps, 1))
        for i in range(0, steps):
            loc = (i + 1) * stepsize
            if (i == steps - 1) & (loc > n - 1):
                threshold[i] = indicator.iloc[n - 1, :]
            else:
                threshold[i] = indicator.iloc[loc, :]
        return threshold

    def combine_state(self, state1, state2, state3):
        return state1 * 100 + state2 * 10 + state3 * 1


    # Implement action and update trade, position ,portfolio value accordingly
    def action_implement(self, cur_cash, index, action, prices, position):
        # buy
        if action == 0:
            shares = 1000 - position
            price = prices.loc[index, self.symbol] * (1 + self.impact)
            if position < 0:
                gain = ((self.prev_price - price) * shares - self.commission) if shares != 0 else 0
            else:
                gain = ((price - self.prev_price) * shares - self.commission) if shares != 0 else 0

            if gain > 0:
                prices.loc[index, "Trade"] = shares
                position += prices.loc[index, "Trade"]
                cash_change = (price * shares - self.commission)
                cur_cash = cur_cash - cash_change
                self.prev_price = price

        # sell
        elif action == 1:
            shares = -1000 - position
            price = prices.loc[index, self.symbol] * (1 - self.impact)
            gain = ((price - self.prev_price) * shares * (-1) - self.commission) if shares != 0 else 0
            if gain > 0:
                prices.loc[index, "Trade"] = shares
                position += prices.loc[index, "Trade"]
                cash_change = (price * shares * (-1) - self.commission)
                cur_cash = cur_cash + cash_change
                self.prev_price = price

        # nothing
        else:
            prices.loc[index, 'Trade'] = 0
            position += prices.loc[index, 'Trade']

        prices.loc[index, "Position"] = position
        prices.loc[index, "stockval"] = prices.loc[index, self.symbol] * position
        prices.loc[index, "cash"] = cur_cash
        prices.loc[index, "porval"] = prices.loc[index, "stockval"] + prices.loc[index, "cash"]
        return cur_cash, position

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol="AAPL",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        """

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]
        prices['prices_percent'] = prices_all[syms].pct_change(1)
        prices['prices_percent'].iloc[0] = 0


        self.symbol = symbol

        rsi = idc.relative_Strength_index(symbol=symbol, sd=sd, ed=ed, lookback=14)
        sti = idc.stochastic_Indicator(symbol=symbol, sd=sd, ed=ed, lookback=20)
        d = sti.rolling(window=3, min_periods=3).mean()
        d.loc[0:2, :] = 0
        cci = idc.commodity_Channel_index(symbol=symbol, sd=sd, ed=ed, lookback=20)

        steps = 10
        rsi_state = self.discretize(rsi, steps)
        d_state = self.discretize(d, steps)
        cci_state = self.discretize(cci, steps)

        # initialize the learner
        self.learner = ql.QLearner(
            num_states=1000,
            num_actions=3,
            alpha=0.2,
            gamma=0.99,
            rar=0.9,
            radr=0.999,
            dyna=0,
            verbose=False,
        )

        # variable to check cumulative return
        prev_cum_return = 0
        cum_return = 0.2
        cum_mode = 0

        used_time = 0
        before_loop = time.time()

        prices['Trade'] = 0
        prices["Position"] = 0
        prices["stockval"] = 0
        prices["cash"] = 0
        prices["porval"] = 0

        # keep monitoring portfolio value: if same portfolio value  show up in consecutive order 10 times
        # it is very high chance, learner is converged now
        # when portfolio value stop updating or time run out, end the learning process
        while cum_mode <= 10 and used_time <= 23:

            if prev_cum_return == cum_return:
                cum_mode += 1
            else:
                cum_mode = 0

            prev_cum_return = cum_return

            # reset prev_price and position these two tracking variable for new round
            self.prev_price = 0
            position = 0

            # first row of data
            iterrow = prices.iterrows()
            first_index, fist_price = next(iterrow)

            rsi_s = np.argmax(rsi_state >= rsi.loc[first_index, symbol])
            d_s = np.argmax(d_state >= d.loc[first_index, symbol])
            cci_s = np.argmax(cci_state >= cci.loc[first_index, symbol])

            s = self.combine_state(cci_s, rsi_s, d_s)  # first state
            a = self.learner.querysetstate(s)  # set the state and get first action
            cur_cash, position = self.action_implement(sv, first_index, a, prices, position) # implement action
            for index, day in iterrow:

                # Compute the current state
                rsi_s = np.argmax(rsi_state >= rsi.loc[index, symbol])
                d_s = np.argmax(d_state >= d.loc[index, symbol])
                cci_s = np.argmax(cci_state >= cci.loc[index, symbol])
                s_prime = self.combine_state(cci_s, rsi_s, d_s)

                # Compute the reward for the last action
                prev_r = prices.loc[index, 'prices_percent'] * position

                # Query the learner with the current state and reward to get an action
                action = self.learner.query(s_prime, prev_r)

                # Implement the action the learner returned (LONG, CASH, SHORT), and update portfolio value
                cur_cash, position = self.action_implement(cur_cash, index, action, prices, position)

            cum_return = (prices["porval"].iloc[-1] / prices['porval'].iloc[0]) - 1
            used_time = time.time() - before_loop

    def testPolicy(
            self,
            symbol="AAPL",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			 	 	 		 		 	
        """
        syms = [symbol]
        self.symbol = symbol
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]
        prices_percent = prices_all[syms].pct_change()
        prices_percent[symbol].iloc[0] = 0

        rsi = idc.relative_Strength_index(symbol=symbol, sd=sd, ed=ed, lookback=14)
        sti = idc.stochastic_Indicator(symbol=symbol, sd=sd, ed=ed, lookback=20)
        d = sti.rolling(window=3, min_periods=3).mean()
        d.loc[0:2, :] = 0
        cci = idc.commodity_Channel_index(symbol=symbol, sd=sd, ed=ed, lookback=20)

        steps = 10
        rsi_state = self.discretize(rsi, steps)
        d_state = self.discretize(d, steps)
        cci_state = self.discretize(cci, steps)

        test_position = 0
        prices['Trade'] = 0
        cur_cash = sv
        for index, day in prices.iterrows():
            rsi_s = np.argmax(rsi_state >= rsi.loc[index, symbol])
            d_s = np.argmax(d_state >= d.loc[index, symbol])
            cci_s = np.argmax(cci_state >= cci.loc[index, symbol])
            state = self.combine_state(cci_s, rsi_s, d_s)
            # find action by querying existing Q Learner Table
            action = self.learner.querysetstate(state)
            cur_cash, test_position = self.action_implement(cur_cash, index, action, prices, test_position)

        Trade = prices["Trade"].copy()
        result = pd.DataFrame({symbol: Trade.values}, index=Trade.index)
        return result
