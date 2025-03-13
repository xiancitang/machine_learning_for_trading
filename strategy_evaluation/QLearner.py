""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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

import random as rand
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_states: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_actions: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type alpha: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type gamma: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type rar: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			 	 	 		 		 	
    :type radr: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type dyna: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "cici"

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar  # decay after  each update
        self.radr = radr
        self.num_states = num_states
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.QTable = np.zeros([num_states, num_actions], dtype=float)

        if dyna > 0:
            self.TC = np.full((num_states, num_actions,num_states), 0.00001, dtype=float)
            self.T = np.zeros([num_states, num_actions, num_states], dtype=float)
            self.R = np.zeros([num_states, num_actions], dtype=float)

    def querysetstate(self, s):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Update the state without updating the Q-table  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param s: The new state  		  	   		  		 			  		 			 	 	 		 		 	
        :type s: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The selected action  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
        """
        a = np.argmax(self.QTable[s, :])

        # update current state and action
        self.s = s
        self.a = a
        return a

    def query(self, s_prime, r):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Update the Q table and return an action  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param s_prime: The new state  		  	   		  		 			  		 			 	 	 		 		 	
        :type s_prime: int  		  	   		  		 			  		 			 	 	 		 		 	
        :param r: The immediate reward  		  	   		  		 			  		 			 	 	 		 		 	
        :type r: float  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The selected action  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
        """

        # dyna section
        if self.dyna > 0:

            # update TC for count of tuple (s,a,s_prime)
            self.TC[self.s, self.a, s_prime] = self.TC[self.s, self.a, s_prime] + 1

            # update model T , R
            count_s_a_sprime = self.TC[self.s, self.a, s_prime]
            count_s_a = self.TC[self.s, self.a, :].sum()
            self.T[self.s, self.a, s_prime] = count_s_a_sprime / count_s_a
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            # hallucinate dyno number of experiences and update Q Table
            count = 0
            all_prev = np.argwhere(self.TC != 0.00001)
            prev_count = all_prev.shape[0]

            while count < self.dyna:
                p = rand.randint(0, prev_count - 1)
                d_s = all_prev[p][0]
                d_a = all_prev[p][1]
                d_s_prime = np.argmax(self.T[d_s, d_a, :])
                d_r = self.R[d_s, d_a]

                d_a_prime = np.argmax(self.QTable[d_s_prime, :])
                self.QTable[d_s, d_a] = (1 - self.alpha) * self.QTable[d_s, d_a] + self.alpha * (
                        d_r + self.gamma * self.QTable[d_s_prime, d_a_prime])
                count = count + 1

        # 1 update Q Table with experience tuple (s, a, s_prime, r)
        a_prime = np.argmax(self.QTable[s_prime, :])
        self.QTable[self.s, self.a] = (1 - self.alpha) * self.QTable[self.s, self.a] + self.alpha * (
                r + self.gamma * self.QTable[s_prime, a_prime])

        # 2 roll dice to decide whether we take random action or not
        randomrate = self.rar
        if rand.uniform(0.0, 1.0) <= randomrate:  # going rogue
            a = rand.randint(0, self.num_actions - 1)  # choose the random direction
        else:
            a = a_prime

        # 3 update current state and action
        self.s = s_prime
        self.a = a
        self.rar = self.rar * self.radr  # update rar

        return a


if __name__ == "__main__":
    pass
