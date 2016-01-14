#!/usr/bin/env python
# -*- coding: utf-8 -*-u

"""
Purpose : Perform q learning to learn to trade a risky asset

"""
# Import packages
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import mpl_toolkits.mplot3d




def get_discretisation(returns, nb_bins):
    """ Get the discretisation for the returns """
    return np.percentile(returns, np.linspace(0, 100, nb_bins))[1:-1]


def geometric_brown(T=10, mu=0.01, sigma=0.1, S0=1, dt=0.01):
    """ Returns a geometric browninan motion"""
    N = int(round(T / dt))
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # standard brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    return S0 * np.exp(X)  # geometric brownian motion


def shift(arr, n):
    """ shift a one ndarray"""
    e = np.empty_like(arr)
    if n >= 0:
        e[:n] = arr[0]
        e[n:] = arr[:-n]
    else:
        e[n:] = arr[0]
        e[:n] = arr[-n:]
    return e


def sharp_ratio(rewards):
    """ Return sharp ratio of returns """
    return np.mean(rewards) / np.std(rewards)


def wealth(rewards):
    return rewards.cumprod(1 + rewards)


def moving_average(rewards):
    """ return average of the t days"""
    return np.mean(rewards)


class QlearningNaive(object):

    def __init__(self, market_returns):
        self.market_returns = market_returns
        self.samples = len(self.market_returns)
        self.actions = np.array([0, 1])  # short long
        self.states = np.array([0, 1])
        self.nb_actions = len(self.actions)
        self.nb_states = 2
        self.epsilon = 0.05
        self.risk_free_rate = 0
        self.transaction_cost = 0
        # self.alpha = 0.05
        self.utility_function = sharp_ratio
        self.gamma = 0.95  # discount factor
        self.nmc = 8  # nb of simulations of Monte Carlo
        self.states = np.zeros(self.samples)
        self.q = np.zeros((self.nb_states, self.nb_actions))
        self.rewards = np.zeros(self.samples)
        self.nb_visits = np.ones((self.nb_states, self.nb_actions))
        self.current_state = self.rewards[1]

    def reset(self):
        self.nb_visits = np.ones((self.nb_states, self.nb_actions))
        self.rewards = np.zeros(self.samples)

    def get_reward(self, action, market_return):
        """ Get reward associated at time t"""
        return market_return * action

#     def update_reward(self, reward, t):
#         """ Update reward at time t"""
#         self.rewards = reward[t]

    def get_action(self, state):
        """ Select actions with epison greedy Q learning """
        # starting point of algorithm
        if (self.q[state, :] == 0).all():
            return np.random.choice(self.actions)
        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q[state, :])

    @staticmethod
    def action_to_trade(action):
        """map position in matrix to trading decision"""
        if action == 0:
            return -1
        if action == 1:
            return 1

    def simu_transition(self, reward):
        """ Returns simulation of next state based on action and previous state """
        if reward >= 0:
            return 1
        if reward < 0:
            return 0

    def qlearning_step(self, verbose=True):
        for i in range(self.samples - 1):
            #market_return = self.market_returns[i]
            a = self.get_action(self.current_state)  # get action
            trading_position = self.action_to_trade(a)
            self.nb_visits[self.current_state, a] += 1  # update nb_visits
            # get immediate reward
            self.reward = self.get_reward(
                trading_position, self.market_returns[i + 1])
            # get new state
            new_state = self.simu_transition(self.reward)
            self.rewards[i] = self.reward  # put reward in rewards
            # Compute temporal difference
            td = self.reward + self.gamma * \
                (self.q[new_state, :].max()) - self.q[self.current_state, a]
            # Print infos
            if verbose:
                print("state : {}, new_state : {}, actions : {}, reward :{}, ".format(
                    self.current_state, new_state, trading_position, self.reward))
                print("Q matrix : \n {}".format(self.q))
            self.q[self.current_state, a] += + \
                (1.0 / self.nb_visits[self.current_state, a]) * td
            self.states[i] = new_state
            self.current_state = new_state

    def monte_carlo(self, verbose=False):
        """ Perform monte carlo iteration for Q learning """
        list_wealth = []
        list_q = []
        for i in range(self.nmc):
            self.qlearning_step(verbose=verbose)
            plt.plot((1 + self.rewards).cumprod(),
                     label="Simulation nb : {}".format(i))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=3, fancybox=True, shadow=True)
            list_wealth.append((1 + self.rewards).cumprod())
            list_q.append(self.q)
            self.reset()  # reset nb_visits and starting point
        return self.q, np.array(list_q), np.array(list_wealth)


class QlearningFunction(object):

    def __init__(self, market_returns):
        self.market_returns = market_returns
        self.samples = len(self.market_returns)
        self.actions = np.array([0, 1, 2])  # short, out of market, long
        #self.states = np.array([0,1])
        self.state_function = moving_average
        self.nb_actions = len(self.actions)
        self.window_short = 10
        self.window_long = 50
        self.nb_states = 8
        self.epsilon = 0.05
        self.risk_free_rate = 0
        self.transaction_cost = 0
        self.gamma = 0.95  # discount factor
        self.nmc = 20  # nb of simulations of Monte Carlo
        self.q = np.zeros((self.nb_states, self.nb_states, self.nb_actions))
        self.rewards = np.zeros(self.samples)
        self.bins_discrete = get_discretisation(
            returns_sp500, self.nb_states - 1)  # discretization
        self.nb_visits = np.ones(
            (self.nb_states, self.nb_states, self.nb_actions))

    def reset(self):
        self.nb_visits = np.zeros((self.nb_states, self.actions))
        self.current_state = self.random.choice([0, 1])

    def get_reward(self, action, market_return):
        """ Get reward """
        return market_return * action  # no transaction cost and risk free rate = 0

    def current_state(self, signal):
        """ Discretization of the the continuous states (moving average, sharp ratio) """
        # nb_states is the number of bins here
        return np.sum(self.bins_discrete < float(signal))

    def get_action(self, state_short, state_long):
        # starting point of algorithm
        if (self.q[state_short, state_long, :] == 0).all():
            return np.random.choice(self.actions)
        else:
            # Epsilon-Greedy
            if np.random.random(1) < self.epsilon:
                return np.random.choice(self.actions)
            else:
                return np.argmax(self.q[state_short, state_long, :])

    @staticmethod
    def action_to_trade(action):
        """map position in matrix to trading decision"""
        if action == 0:
            return -1
        if action == 1:
            return 0
        if action == 2:
            return 1

    def qlearning_step(self, verbose=True):
        for i in range(self.samples - 1):
            if i <= self.window_long - 1:
                self.rewards[i] = self.market_returns[i]
            # get actual state
            # state = state_function of returns (moving average, sharp ratio
            # ...)
            state_short = self.current_state(self.state_function(
                self.rewards[i - self.window_short: i]))
            state_long = self.current_state(self.state_function(
                self.rewards[i - self.window_short: i]))
            # get action with Q learning and greedy policy
            a = self.get_action(state_short, state_long)
            trading_position = self.action_to_trade(a)
            self.nb_visits[state_short, state_long, a] += 1  # update nb_visits
            # get immediate reward
            self.reward = self.get_reward(
                trading_position, self.market_returns[i])
            self.rewards[i] = self.reward  # update reward
            # get new states
            new_state_short = self.current_state(self.state_function(
                self.rewards[i - self.window_short: i + 1]))
            new_state_long = self.current_state(self.state_function(
                self.rewards[i - self.window_short: i + 1]))
            # Compute temporal difference
            td = self.reward + self.gamma * \
                (self.q[new_state_short, new_state_long:].max()) - \
                self.q[state_short, state_long, a]
            # Print infos
            if verbose:
                print("state : {}, new_state : {}, actions : {}, reward :{}, ".format(
                    (state_short, state_long), (new_state_short, new_state_long), a, self.reward))
                #print("Q matrix : \n {}".format(self.q))
            self.q[state_short, state_long, a] += + \
                (1.0 / self.nb_visits[state_short, state_long, a]) * td

    def monte_carlo(self, verbose=False):
        """ Perform monte carlo iteration for Q learning """
        for i in range(self.nmc):
            print("Simulation nb : {}".format(i))
            self.qlearning_step(verbose=verbose)
            print(1 + model.rewards[1:]).cumprod()
            self.reset()  # reset nb_visits and starting point
            return q


class Qlearning2state(object):

    def __init__(self, market_returns, nb_states):
        self.market_returns = market_returns
        self.samples = len(self.market_returns)
        self.actions = np.array([0, 1, 2])  # short, out of market, long
        self.state_function = moving_average
        self.nb_actions = len(self.actions)
        self.nb_states = nb_states
        self.epsilon = 0.05
        self.risk_free_rate = 0
        self.transaction_cost = 0
        self.gamma = 0.95  # discount factor
        self.nmc = 10  # nb of simulations of Monte Carlo
        self.q = np.zeros((self.nb_states, self.nb_states, self.nb_actions))
        self.rewards = np.zeros(self.samples)
        self.bins_discrete = get_discretisation(
            returns_sp500, self.nb_states + 1)  # discretization
        self.nb_visits = np.ones(
            (self.nb_states, self.nb_states, self.nb_actions))

    def reset(self):
        self.nb_visits = np.ones(
            (self.nb_states, self.nb_states, self.nb_actions))
        self.rewards = np.zeros(self.samples)

    def get_reward(self, action, market_return):
        """ Get reward """
        return market_return * action  # no transaction cost and risk free rate = 0

    def current_state(self, signal):
        """ Discretization of the the continuous states (moving average, sharp ratio) """
        # nb_states is the number of bins here
        return np.sum(self.bins_discrete < float(signal))

    def get_action(self, state_short, state_long):
        # starting point of algorithm
        if (self.q[state_short, state_long, :] == 0).all():
            return np.random.choice(self.actions)
        else:
            # Epsilon-Greedy
            if np.random.random(1) < self.epsilon:
                return np.random.choice(self.actions)
            else:
                return np.argmax(self.q[state_short, state_long, :])

    @staticmethod
    def action_to_trade(action):
        """map position in matrix to trading decision"""
        if action == 0:
            return -1
        if action == 1:
            return 0
        if action == 2:
            return 1

    def qlearning_step(self, verbose=True):
        for i in range(1, self.samples - 1):
            if i <= 1:
                self.rewards[i] = self.market_returns[i]
            # state = state_function of returns (moving average, sharp ratio
            # ...)
            state_short = self.current_state(self.rewards[i])
            state_long = self.current_state(self.rewards[i - 1])
            # get action with Q learning and greedy policy
            a = self.get_action(state_short, state_long)
            trading_position = self.action_to_trade(a)
            self.nb_visits[state_short, state_long, a] += 1  # update nb_visits
            # get reward
            self.reward = self.get_reward(
                trading_position, self.market_returns[i + 1])
            self.rewards[i + 1] = self.reward  # update reward
            # get new states
            new_state_short = self.current_state(self.rewards[i])
            new_state_long = self.current_state(self.rewards[i + 1])
            # Compute temporal difference
            td = self.reward + self.gamma * \
                (self.q[new_state_short, new_state_long:].max()) - \
                self.q[state_short, state_long, a]
            # Print infos
            if verbose:
                print("state : {}, new_state : {}, actions : {}, reward :{},td :{} ".format(
                    (state_short, state_long), (new_state_short, new_state_long), a, self.reward, td))
                #print("Q matrix : \n {}".format(self.q))
            self.q[state_short, state_long, a] += + \
                (1.0 / self.nb_visits[state_short, state_long, a]) * td

    def monte_carlo(self, verbose=False, plot=False):
        """ Perform monte carlo iteration for Q learning """
        list_wealth = []
        list_q = []
        for i in range(self.nmc):
            self.qlearning_step(verbose=verbose)
            if plot:
                plt.plot((1 + self.rewards).cumprod(),
                         label="Simulation nb : {}".format(i))
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                           ncol=3, fancybox=True, shadow=True)
            list_wealth.append((1 + self.rewards).cumprod())
            list_q.append(self.q)
            self.reset()  # reset nb_visits and starting point
        return self.q, np.array(list_q), np.array(list_wealth)

def plot_Q_matrix_3d(Q, nb_states, nb_actions):
    """Credit to nekopuni"""
    x = np.linspace(0, nb_states - 1, nb_states)
    y = np.linspace(0, nb_states - 1, nb_states)
    x, y = np.meshgrid(x, y)
    for i in range(nb_actions):
        if i == 0:
            position = "short"
        elif i == 1:
            position = "flat"
        elif i == 2:
            position = "long"
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_xlabel("state_short")
        ax.set_ylabel("state_long")
        ax.set_zlabel("Q-value")
        ax.set_title("Q-value for " + position + " position")
        #ax.view_init(90, 90)
        urf = ax.plot_surface(x, y, model2_state.q[
                              :, :, i], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()
