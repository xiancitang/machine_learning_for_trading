""""""

"""MC2-P1: Market simulator.  		  	   		  		 			  		 			 	 	 		 		 	

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

import pandas as pd
from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cici"


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		  		 			  		 			 	 	 		 		 	

    :param orders_df:  the trades
    :type orders_df: data frame
    :param start_val: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		  		 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		  		 			  		 			 	 	 		 		 	
    """
    orders_df.sort_index()
    symbols = orders_df.columns
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    # Create dataframe Prices
    # include SPY to make sure all trading days between start day and end date inclusively are included
    Prices_W_SPY = get_data(symbols, pd.date_range(start_date, end_date))
    Prices = Prices_W_SPY.loc[:, Prices_W_SPY.columns != 'SPY']
    Prices['Cash'] = 1

    # Create dataframe Trades
    Trades = pd.DataFrame(data=0, columns=Prices.columns, index=Prices.index)

    for index, order in orders_df.iterrows():
        sym = symbols[0]
        ty = "BUY" if order[sym]>0 else "SELL" if order[sym]<0 else "NOTHING"
        shares = order[sym]
        price = Prices.loc[index, sym] * (1 + impact) if ty == "BUY" else Prices.loc[index, sym] * (1 - impact)
        # If order happens on valid trading day, update dataframe Trades
        if index in Prices.index:
            Trades.loc[index, sym] = Trades.loc[index, sym] + shares
            Trades.loc[index, "Cash"] = Trades.loc[index, "Cash"] + shares * price * (-1) - commission

    # Create dataframe Holdings
    Temp = Trades.loc[:, Trades.columns != 'Cash']
    Temp_Sum = Temp.cumsum()
    Holdings = pd.concat([Temp_Sum, Trades['Cash']], axis=1)
    Holdings.iloc[0, Holdings.columns.get_loc("Cash")] = start_val + Holdings["Cash"].iloc[0]
    Holdings["Cash"] = Holdings["Cash"].cumsum()

    # Create dataframe Values
    Values = Prices * Holdings
    Values['Port_Value'] = Values.sum(axis=1)
    portvals = Values['Port_Value']

    return portvals
