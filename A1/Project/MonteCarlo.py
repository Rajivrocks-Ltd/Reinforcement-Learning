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


class MonteCarloAgent(BaseAgent):

    def update(self, states, actions, rewards):
        """ states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state """
        # TO DO: Add own code

        T = len(states) - 1  # Total number of time steps in the episode

        G = 0  # Initialize the infinite sum return

        for t in reversed(range(T)):
            G = self.gamma * G + rewards[t]  # Calculate the infinite sum

            # Update the Q-value for the state-action pair using the absolute difference
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    """ runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep """

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Monte Carlo RL algorithm here!

    for t in range(n_timesteps):
        s = env.reset()
        total_reward = 0
        states = [s]
        actions = []
        rewards = []
        done = False

        for _ in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            states.append(s_next)
            actions.append(a)
            rewards.append(r)

            total_reward += r

            if done:
                break
            else:
                s = s_next

        pi.update(states, actions, rewards)

        if t % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_timesteps.append(t)
            eval_returns.append(eval_return)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)  # Plot the Q-value estimates during Monte
        # Carlo RL execution

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 10000
    max_episode_length = 100
    eval_interval = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                                               policy, epsilon, temp, plot, eval_interval)

    print(eval_returns, eval_timesteps)


if __name__ == '__main__':
    test()
