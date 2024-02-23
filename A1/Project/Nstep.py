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


class NstepQLearningAgent(BaseAgent):

    def update(self, states, actions, rewards, done, n):
        """ states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state """
        # TO DO: Add own code

        T_ep = len(states)  # Total number of time steps in the episode

        for t in range(T_ep):
            m = min(n, T_ep - t)  # m is the number of rewards left to sum

            if done:
                G_t = np.sum([self.gamma ** i * rewards[t + i] for i in range(m - 1)])  # n-step target without
                # bootstrap
            else:
                G_t = np.sum([self.gamma ** i * rewards[t + i] for i in
                              range(m - 1)])  # Exclude the last reward in the bootstrap
                G_t += self.gamma ** m * np.max(self.Q_sa[states[t + m], :])

            # Update the Q-value for the current state-action pair in the episode using the absolute difference
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G_t - self.Q_sa[states[t], actions[t]])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    """ runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep """

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    # rows, cols = pi.Q_sa.shape
    # pi.Q_sa = np.zeros((rows, cols))

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

        pi.update(states, actions, rewards, done, n)

        if t % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_timesteps.append(t)
            eval_returns.append(eval_return)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)  # Plot the Q-value estimates during n-step
        # Q-learning execution

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 100000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    eval_return, eval_timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                                           policy, epsilon, temp, plot, n=n)

    print(eval_return, eval_timesteps)


if __name__ == '__main__':
    test()
