
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1/self.N) * ((self.N-1)*(self.mean) + x)


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumavg = np.cumsum(data)/(np.arange(N)+1)

    for b in bandits:
        print(b.mean)

    return cumavg


if __name__ == '__main__':
    c_1 = run_experiment(1., 2., 3., 0.1, 100000)
    c_05 = run_experiment(1., 2., 3., 0.05, 100000)
    c_01 = run_experiment(1., 2., 3., 0.01, 100000)

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()
