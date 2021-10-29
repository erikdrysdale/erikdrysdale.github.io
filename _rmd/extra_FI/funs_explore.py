import os
import numpy as np
from scipy.stats import norm

"""
s:          # of successes
n:          # of trials
alpha:      Type-I error rate
Returns adjusted probabilities and sample sizes
"""
def agresti_trans(s, n, alpha):
    q_a2 = norm.ppf(1-alpha)**2
    ntil = n + q_a2
    ptil = (s + q_a2/2)/ntil
    return ptil, ntil


# Function to save plotnine ggplot
def gg_save(fn,fold,gg,width,height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height, limitsize=False)


#  Define a binomial experiment class for equal sample sizes
class binom_experiment():
    def __init__(self, n, pi1, pi2=None, alpha=0.05, agresti=True):
        self.n = n
        self.alpha = alpha
        self.agresti = agresti
        self.pi1 = pi1
        self.pi2 = pi2
        if self.pi2 is None:
            self.pi2 = self.pi1
        self.pid = self.pi2 - self.pi1
        self.null = True
        if self.pid > 0:
            self.null = False
        # Calculate the oracle power
        self.q_alpha = norm.ppf(1-alpha)
        num = np.sqrt(2*self.pi1*(1-self.pi1))*self.q_alpha-np.sqrt(n)*self.pid
        den = np.sqrt(self.pid + self.pi1*(2-self.pi1) - (self.pi1 + self.pid)**2 )
        self.power = 1 - norm.cdf(num / den)
        
    def dgp(self, nsamp, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Draw data
        self.s1 = np.random.binomial(self.n, self.pi1, nsamp)
        self.s2 = np.random.binomial(self.n, self.pi2, nsamp)
        # Calculate sample statistic and variance
        if self.agresti:
            n_tilde = self.n + self.q_alpha**2
            self.pi1_hat = (self.s1 + 0.5*self.q_alpha**2) / n_tilde
            self.pi2_hat = (self.s2 + 0.5*self.q_alpha**2) / n_tilde
        else:
            self.pi1_hat = self.s1 / self.n
            self.pi2_hat = self.s2 / self.n
        self.pid_hat = self.pi2_hat - self.pi1_hat
        self.pi_hat = (self.pi1_hat + self.pi2_hat) / 2
        # Calculate statistic under the null
        self.se_null = np.sqrt( 2 * self.pi_hat*(1-self.pi_hat) / self.n )
        self.z = self.pid_hat/self.se_null
        self.pval = 1 - norm.cdf(self.z)
        self.reject = np.where(self.pval < self.alpha, 1, 0)

