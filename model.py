import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class ThompsonSampler:
    def __init__(self, a0, b0, x, r, lr=1, N=10000):
        """
        :param a0 (ndarray: [K,]): initialization of alpha parameters for each arm
        :param b0 (ndarray: [K,]): initialization of beta parameters for each arm
        :param x  (ndarray: [T,]): history of choices
        :param r  (ndarray: [T,]): history of rewards
        :param lr (ndarray: [2,]): learning rates
        :param N  (int)          : number of times to sample the posterior distribution on each iteration
        """
        self.K = len(a0)
        self.N = N
        self.lr = lr

        self.a = a0
        self.b = b0
        self.x = x
        self.r = r

        self.a_history = []
        self.b_history = []
        self.loss_history = []
        self.posterior_history = []

    def fit(self):
        """
        dummy call
        """
        for t in range(len(self.x)):
            self.step(t)
        return

    def step(self, t):
        """
        - sample posterior probability
        - update loss, append to loss_history
        - update alpha, beta
        """
        posterior_samples = self.posterior()
        self.loss_history.append(1 -(posterior_samples[self.x[t]]))
        self.posterior_history.append(posterior_samples)
        self.update(self.r[t], self.x[t])
        return

    def posterior(self):
        """
        - sample from the posterior distribution
            - generate N values from the beta distribution for each arm
            - for each iter, return the arm with the largest sampled value
            - return the posterior probability (proportion of samples) for each arm k
        """
        samples = np.array([beta.rvs(a_, b_, size=self.N)
                            for a_, b_ in zip(self.a, self.b)])
        sample_arm = np.argmax(samples, axis=0)
        return np.array([sum(sample_arm == arm) for arm in range(self.K)])/self.N

    def update(self, r_t, x_t):
        """
        update the alpha and beta values for arm 'x' with the reward outcome at time t
        append the updated alpha and beta values to history
        """
        self.a_history.append(self.a.copy())
        self.b_history.append(self.b.copy())
        self.a[x_t] = self.a[x_t] + self.lr[0]*r_t
        self.b[x_t] = self.b[x_t] + self.lr[1]*(1 - r_t)
        return


if __name__ == "__main__":
    import inspect
    a_ = [1, 1, 1, 1]
    b_ = [1, 1, 1, 1]
    x_ = [1, 2]
    r_ = [0, 1]
    model = ThompsonSampler(a_, b_, x_, r_)
    model.step(0)
    model.step(1)
    for attr in inspect.getmembers(model):
        if not attr[0].startswith('_'):
            print(attr)
