import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
# EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class OptimisticBandit:
    def __init__(self, p, opt_mean):
        self.p = p
        self.mean =opt_mean
        self.N = 1  # this has to be set to 1, not 0 otherwise it is erased as soon as the update function is called
        
    def pull(self):
        return np.random.random() < self.p 
    
    def update(self, x):
        self.N += 1
        self.mean = ((self.N-1)*self.mean+x)/self.N
        
def experiment():
    bandits = [OptimisticBandit(p, 5) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
    
    for b in bandits:
        print(f'mean estimate: {b.mean}')
        
        # print total reward
    total_rewards = rewards.sum()
    print(f'total reward earns: {total_rewards}')
    print(f'overall win rate: {total_rewards/NUM_TRIALS}')
    print(f'num times selected each bandit:{[b.N for b in bandits]}')
    
    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()
    
if __name__ == '__main__':
    experiment()
    