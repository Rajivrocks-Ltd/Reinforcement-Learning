#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


class SarsaAgent(BaseAgent):

    def update(self, s, a, r, s_next, a_next, done):
        # TO DO: Add own code
        if done:
            self.Q_sa[s, a] = r
        else:
            g = r + self.gamma * self.Q_sa[s_next, a_next]
            self.Q_sa[s, a] += self.learning_rate * (g - self.Q_sa[s, a])


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    """ runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep """

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!
    for t in range(n_timesteps):
        s = env.reset()
        a = pi.select_action(s, policy, epsilon, temp)

        total_reward = 0

        while True:
            s_next, r, done = env.step(a)
            a_next = pi.select_action(s_next, policy, epsilon, temp)
            pi.update(s, a, r, s_next, a_next, done)

            total_reward += r

            if done:
                break

            s = s_next
            a = a_next

        # Evaluation at regular intervals
        if t % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_timesteps.append(t)
            eval_returns.append(eval_return)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                   step_pause=0.1)  # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    eval_interval = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)

    print(eval_returns, eval_timesteps)


R N R R N N R N Nif __name__ == '__main__':
    test()
