import os
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.stats import multivariate_normal as MVN
from scipy.linalg import cholesky
from scipy.optimize import minimize_scalar
from plotnine import *

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'figures')

###########################################
# --- COMPARE BVN INTEGRATION METHODS --- #

# https://stats.stackexchange.com/questions/61080/how-can-i-calculate-int-infty-infty-phi-left-fracw-ab-right-phiw?noredirect=1&lq=1
# https://math.stackexchange.com/questions/449875/expected-value-of-normal-cdf/1125935
# https://math.stackexchange.com/questions/2392827/normal-random-variable-as-argument-of-standard-normal-cdf


def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return rvec(x).T


class BVN():
    # mu=np.array([1,2]);sigma=np.array([2,3]); rho=0.5; # del mu, sigma, rho, seed
    def __init__(self, mu, sigma, rho):
        """
        mu: array of means
        sigma: array of variances
        rho: correlation coefficient
        """
        assert mu.shape[0]==sigma.shape[0]==2
        assert np.abs(rho) <= 1
        self.mu = mu.reshape([1,2])
        self.sigma = sigma.flatten()
        od = rho*np.sqrt(sigma.prod())
        self.rho = rho
        self.Sigma = np.array([[sigma[0],od],[od, sigma[1]]])
        self.A = cholesky(Sigma) # A.T.dot(A) = Sigma

    # size=1000;seed=1234 # del size, seed
    def rvs(self, size, seed=None):
        """
        size: number of samples to simulate
        seed: to pass onto np.random.seed
        """
        # size: 
        np.random.seed(seed)
        X = np.random.randn(size,2)
        Z = A.T.dot(X.T).T + mu
        return Z

    # self=dist_BVN; h, k = 2, 4; # del h, k
    def orthant(self, h, k, method='scipy'):
        """
        P(X1 >= h, X2 >=k)
        """
        assert (h > 0) & (k > 0)
        # Calculate the number of standard deviations away it is
        y1, y2 = (np.array([h,k])-self.mu.flat)/np.sqrt(self.sigma)
        
        # (i) scipy: L(h, k)=1-(F1(h)+F2(k))+F12(h, k)
        if method == 'scipy'
            sp_bvn = MVN([0, 0],[[1,self.rho],[self.rho,1]])
            pval_sp = 1+sp_bvn.cdf([y1, y2])-(norm.cdf(y1)+norm.cdf(y2))
        

        if method == 'cox1':
            norm.cdf(-Ah)*norm.cdf((rho*norm.pdf(Ah)/norm.cdf(-Ah) - Ak)/np.sqrt(1-rho**2))

        # https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
        


dist_BVN = BVN(mu=np.array([1,2]),sigma=np.array([0.5,2]),rho=0.4)
qq = dist_BVN.rvs(nsim)
np.mean((qq[:,0] <= h) & (qq[:,1] <= k))


# Runtime & comp of
# i) scipy, ii) Cox approximation (3)+(4), iii) integration of (2)+(3)

lhk3 = norm.cdf(-Ah)*norm.cdf((rho*norm.pdf(Ah)/norm.cdf(-Ah) - Ak)/np.sqrt(1-rho**2))


print('Orthant prob estimates: scipy=%0.4f, rvs=%0.4f, Cox1=%0.4f' % (lhk1,lhk2,lhk3))


gg_bvn = (ggplot(df,aes('X','Y')) + theme_bw() + 
    geom_point(size=0.25,alpha=0.25) + 
    geom_hline(yintercept=k) + geom_vline(xintercept=h))
gg_bvn.save(os.path.join(dir_figures,'gg_bvn.png'),width=5,height=4)


########################
# --- QUERIES 1964 --- #
# X~N(100,6), Y~TN(50,3,44,Inf)

class NT_dist():
    def __init__(self, mu1, se1, mu2, se2, a2, b2):
        # Parameters
        self.mu1 = mu1
        self.se1 = se1
        self.mu2 = mu2
        self.se2 = se2
        self.a2 = a2
        self.b2 = b2
        # Truncated normal
        self.A2 = (a2 - mu2)/se2
        self.B2 = (b2 - mu2)/se2
        self.d2 = norm.pdf(self.A2) - norm.pdf(self.B2)
        self.p2 = norm.cdf(self.B2) - norm.cdf(self.A2)
        self.mean2 = self.mu2 + self.d2/self.p2*se2
        # Distributions
        self.dist1 = norm(loc=mu1, scale=se1)
        self.dist2 = truncnorm(a=self.A2, b=self.B2, loc=mu2, scale=se2)
        self.mean3 = self.mu1 + self.mean2
        # Convert N(Z1), TN(Z2) -> Z1+Z2 MVTN(theta,Sigma,a,b)
        # X = Z1 + Z2, Y=Z2
        self.theta1 = mu1 + mu2
        self.theta2 = mu2
        self.sigma1 = np.sqrt(se1**2 + se2**2)
        self.sigma2 = se2
        self.rho = self.sigma2/self.sigma1


    def pdf(x):
        # (2.2) from Lee
        norm.pdf()


    def rvs(self, n, seed=1234):
        r1 = self.dist1.rvs(size=n,random_state=seed)
        r2 = self.dist2.rvs(size=n,random_state=seed)
        return r1 + r2 
#

mu_x, se_x = 100, 6
mu_y, se_y, a_y, b_y = 50, 3, 49, np.Inf
dist = NT_dist(mu1=mu_x,se1=se_x,mu2=mu_y,se2=se_y,a2=a_y,b2=b_y)
z = dist.rvs(n=100000,seed=1)
t = mu_x+mu_y-1
np.mean(z < t)
(dist.dist1.cdf(t*(mu_x/(mu_y+mu_x)))+dist.dist2.cdf(t*(mu_y/(mu_y+mu_x))))/2








        self.mean = self.mu1 + self.mu2

    
    def cdf_w(self, w):
        a = np.sqrt(1+self.r**2) * self.k * self.s
        b = -self.r * self.s
        rho = -b/np.sqrt(1+b**2)
        Sigma = np.array([[1,rho],[rho,1]])
        dist_MVN = MVN(mean=np.repeat(0,2),cov=Sigma)
        x1 = a / np.sqrt(1+b**2)
        if isinstance(w, float):
            X = [x1, w]
        else:
            X = np.c_[np.repeat(x1,len(w)), w]
        pval = dist_MVN.cdf(X)
        return pval
    
    def cdf_x(self, x):
        const = 1 / norm.cdf(self.s * self.k)
        w = (x + self.r * self.k) / np.sqrt(1+self.r**2)
        pval = self.cdf_w(w) * const
        return pval
    
    def quantile(self, p):
        res = minimize_scalar(fun=lambda x: (self.cdf_x(x)-p)**2, method='brent').x
        return res