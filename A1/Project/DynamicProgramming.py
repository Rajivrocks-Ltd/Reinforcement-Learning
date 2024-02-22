#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    """ Class to store the Q-value iteration solution, perform updates, and select the greedy action """

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.threshold = threshold
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        """ Returns the greedy best action in state s """
        # TO DO: Add own code
        # a = np.random.randint(0, self.n_actions)  # Replace this with correct action selection
        a = argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        """ Function updates Q(s,a) using p_sas and r_sas """
        # TO DO: Add own code
        Q_sa_new = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))
        # self.Q_sa = Q_sa_new
        return Q_sa_new


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    """ Runs Q-value iteration. Returns a converged QValueIterationAgent object """

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE

    # Implement the q-value iteration algorithm given the codebase above
    # The algorithm should run until the maximum change in Q-values is less than the threshold

    max_error = np.inf
    i = 0
    while max_error > threshold:
        max_error = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s, a)
                Q_sa_new = QIagent.update(s, a, p_sas, r_sas)
                max_error = max(max_error, abs(Q_sa_new - QIagent.Q_sa[s, a]))
                QIagent.Q_sa[s, a] = Q_sa_new
        i += 1

    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))


if __name__ == '__main__':
    experiment()
