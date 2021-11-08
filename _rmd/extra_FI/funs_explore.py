import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar

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



# Function to find power
def power_func(power, target):
    ipower = norm.ppf(power)
    return (ipower + norm.pdf(-ipower)/norm.cdf(ipower) - target)**2

def find_power(lhs, nprint):
    N = len(lhs)
    holder = np.zeros(N)
    for i in range(N):
        if (i+1) % nprint == 0:
            print('Iteration %i of %i' % (i+1, N))
        res = minimize_scalar(power_func,method='bounded',bounds=(0,1), args=(lhs[i])).x
        holder[i] = res
    return holder

#  Define a binomial experiment class for equal sample sizes
class BPFI():
    def __init__(self, n, pi1, pi2=None, alpha=0.05):
        self.n = n
        self.alpha = alpha
        self.pi1 = pi1
        self.pi2 = pi2
        if self.pi2 is None:
            self.pi2 = self.pi1
        self.pid = self.pi2 - self.pi1
        self.pi0 = (self.pi1 + self.pi2) / 2
        self.null = True
        if self.pid > 0:
            self.null = False
        # Calculate the oracle power
        self.t_a = norm.ppf(1-alpha)
        num = np.sqrt(2*self.pi0*(1-self.pi0))*self.t_a-np.sqrt(n)*self.pid
        den = np.sqrt(self.pid + self.pi1*(2-self.pi1) - (self.pi1 + self.pid)**2 )
        self.power = 1 - norm.cdf(num / den)

        # Calculate the expected value of the FI
        se_f = np.sqrt(2*self.n*self.pi0*(1-self.pi0))
        self.mu_f = self.n*self.pid - self.t_a*se_f
        self.mu_f += se_f*norm.pdf(-norm.ppf(self.power))/norm.cdf(norm.ppf(self.power))


    @staticmethod
    def pval_bp(s1, s2, n):
        pid_hat = (s2 - s1)/n
        pi0_hat = (s2 + s1) / (2*n)
        se_null = np.sqrt( 2 * pi0_hat*(1-pi0_hat) / n )
        z = pid_hat/se_null
        pval = 1 - norm.cdf(z)
        return pval

    def dgp(self, nsamp, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Draw data
        self.s1 = np.random.binomial(self.n, self.pi1, nsamp)
        self.s2 = np.random.binomial(self.n, self.pi2, nsamp)
        self.pi0_hat = (self.s2 + self.s1) / (2*self.n)
        self.se_null = np.sqrt( 2 * self.pi0_hat*(1-self.pi0_hat) / self.n )
        self.pval = BPFI.pval_bp(self.s1, self.s2, self.n)
        self.reject = self.pval < self.alpha

    def fi(self, ncheck=1000):
        # Calculate the FI
        a = (2*self.n+self.t_a**2)
        b = 2*(self.t_a**2*(self.s1-self.n)-2*self.n*self.s1)
        c = self.s1*(2*self.n*self.s1+self.t_a**2*(self.s1-2*self.n))
        s2hat1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        fi1 = self.s2 - s2hat1

        # Calculation FI(approx)
        s2hat2 = (self.s1 + np.sqrt(2*self.n*self.pi0_hat*(1-self.pi0_hat))*self.t_a)
        fi2 = self.s2 - s2hat2
        
        # Inverse calculation of power
        pi1hat = self.s1/self.n
        pi2hat = self.s2/self.n
        sig_hat = np.sqrt(self.n*(pi1hat*(1-pi1hat) + pi2hat*(1-pi2hat)))
        lhs = fi2/sig_hat
        lhs_pos = lhs[self.reject]
        n_pos = len(lhs_pos)
        if n_pos > ncheck:
            n_samp = min(n_pos, ncheck)
            lhs_pos = pd.Series(lhs_pos).sample(n=n_samp,replace=False,random_state=n_samp).values
            self.power_mu = find_power(lhs=lhs_pos, nprint=int(n_samp/5))

        # Store
        self.df_fi = pd.DataFrame({'reject':self.reject,'fi':fi1, 'fia':fi2})
        
