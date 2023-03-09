from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N


def ucb1(mean, n, n_j):
    num = np.log(n)
    j = mean + np.sqrt(2*num/n_j)
    return j


def run_experiment():
    # initialize bandits
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)
    total_plays = 0

    # initalize each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    # play bandits
    for i in range(NUM_TRIALS):
        j = np.argmax([ucb1(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for plots
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.p_estimate)

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])

    return cumulative_average


if __name__ == '__main__':
    run_experiment()
