import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        '''
        p: win rate
        '''
        self.p = p
        self.p_estimate = 0.
        self.N = 0.  # number of samples collected so far

    def pull(self):
        # draw a 1 with a probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate + x) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    # initialize rewards, number of times explored, number of times exploited
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0

    # get optimal j
    optimal_j = np.argmax(b.p for b in bandits)
    print(f'optimal j: {optimal_j}')

    # epsilon greedy
    for i in range(NUM_TRIALS):
        if np.random.random() < EPSILON:
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update the reward
        rewards[i] = x

        bandits[j].update(x)

    # print mean estimates for each bandit
    for b in bandits:
        print(f'p_estimate: {b.p_estimate}')

    # print total reward
    total_rewards = rewards.sum()
    print(f'total reward earns: {total_rewards}')
    print(f'overall win rate: {total_rewards/NUM_TRIALS}')
    print(f'num times explored: {num_times_explored}')
    print(f'num times exploited: {num_times_exploited}')
    print(f'num optimal bandits: {num_optimal}')

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == '__main__':
    experiment()
