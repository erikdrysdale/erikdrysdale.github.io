import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, t, chi2
from scipy.stats import multivariate_normal as MVN
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# WRITE A FUNCTION WRAPPER TO GENERATE DATA
class two_stage():
    def __init__(self, n, m, gamma, alpha, pool=True, student=True):
        # Assign
        assert (n > 1) and (m >= 1) and (gamma > 0) & (gamma < 1)
        self.n, self.m, self.gamma = n, m, gamma
        self.alpha, self.pool = alpha, pool

        # Calculate degres of freedom
        self.dof_S, self.dof_T = n - 1, m - 1
        if self.pool:
            self.dof_T = n + m - 1
        if student:
            self.phi_inv = t(df=self.dof_S).ppf(1-gamma)
        else:
            self.phi_inv = norm.ppf(1-gamma)
        mn_ratio = m / n
        mu_2stage = np.array([0, -np.sqrt(mn_ratio)*self.phi_inv])
        tau_2stage = np.sqrt([1, mn_ratio])
        self.H0 = NTS(mu=mu_2stage,tau=tau_2stage, a=0, b=np.infty)
        self.HA = NTS(mu=mu_2stage,tau=tau_2stage, a=-np.infty, b=0)
        self.t_alpha = self.H0.ppf(alpha)[0]
        self.power = self.HA.cdf(self.t_alpha)

    # self = dist_2s; nsim=100000; delta=2; sigma2=4; seed=None
    def rvs(self, nsim, delta, sigma2, seed=None):
        if seed is None:
            seed = nsim
        np.random.seed(seed)
        delta1 = delta + np.sqrt(sigma2/self.n)*np.random.randn(nsim)
        delta2 = delta + np.sqrt(sigma2/self.m)*np.random.randn(nsim)
        sigS = np.sqrt(sigma2*chi2(df=self.dof_S).rvs(nsim)/self.dof_S)
        sigT = np.sqrt(sigma2*chi2(df=self.dof_T).rvs(nsim)/self.dof_T)
        delta0 = delta1 + (sigS/np.sqrt(self.n))*self.phi_inv
        shat = (delta2 - delta0)/(sigT/np.sqrt(self.m))
        df = pd.DataFrame({'shat':shat, 'd0hat':delta0})
        return df

# self = NTS(mu=np.array([0.5,0.75]),tau=np.array([1,3]), a=0, b=3)
class NTS():
    def __init__(self, mu, tau, a, b):
        """
        mu: array of means
        tau: array of standard errors
        rho: correlation coefficient
        """
        assert mu.shape[0]==tau.shape[0]==2
        # Assign parameters
        self.mu, self.tau = mu.flatten(), tau.flatten()
        self.a, self.b = a, b
        # Truncated normal (Z2)
        self.alpha = (self.a - self.mu[1]) / self.tau[1]
        self.beta = (self.b - self.mu[1]) / self.tau[1]
        self.Z = norm.cdf(self.beta) - norm.cdf(self.alpha)
        self.Q = norm.pdf(self.alpha) - norm.pdf(self.beta)
        # Average will be unweighted combination of the two distributions
        self.mu_W = self.mu[0] + self.mu[1] + self.tau[1]*self.Q/self.Z
        # Distributions
        self.dist_X1 = norm(loc=self.mu[0], scale=self.tau[0])
        self.dist_X2 = truncnorm(a=self.alpha, b=self.beta, loc=self.mu[0], scale=self.tau[1])
        # W
        self.theta1 = self.mu.sum()
        self.theta2 = self.mu[1]
        self.sigma1 = np.sqrt(np.sum(self.tau**2))
        self.sigma2 = self.tau[1]
        self.rho = self.sigma2/self.sigma1

    def pdf(self, x):
        term1 = self.sigma1 * self.Z
        m1 = (x - self.theta1) / self.sigma1
        term2 = (self.beta-self.rho*m1)/np.sqrt(1-self.rho**2)
        term3 = (self.alpha-self.rho*m1)/np.sqrt(1-self.rho**2)
        f = norm.pdf(m1)*(norm.cdf(term2) - norm.cdf(term3)) / term1
        return f

    def cdf(self, x, method='scipy'):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        nx = len(x)
        m1 = (x - self.theta1) / self.sigma1
        bvn = BVN(mu=[0,0],sigma=[1,1],rho=self.rho)
        alpha_seq = np.repeat(self.alpha,nx)
        alpha_seq = np.where(alpha_seq == -np.infty, -10, alpha_seq)
        beta_seq = np.repeat(self.beta,nx)
        beta_seq = np.where(beta_seq == np.infty, +10, beta_seq)
        orthant1 = bvn.orthant(m1,alpha_seq,method=method)
        orthant2 = bvn.orthant(m1,beta_seq,method=method)
        orthant = 1 - (orthant1 - orthant2)/self.Z
        return orthant

    def ppf(self, p):
        if isinstance(p, list):
            p = np.array(p)
        if isinstance(p, float) or isinstance(p, int):
            p = np.array([p])
        # Set up reasonable lower bounds
        lb = self.mu_W - self.sigma1*4
        ub = self.mu_W + self.sigma1*4
        w = np.repeat(np.NaN, len(p))
        for i, px in enumerate(p):
            tmp = float(minimize_scalar(fun=lambda w: (self.cdf(w)-px)**2,method='bounded',bounds=(lb,ub)).x)
            w[i] = tmp
        return w

    def rvs(self, n, seed=1234):
        r1 = self.dist_X1.rvs(size=n,random_state=seed)
        r2 = self.dist_X2.rvs(size=n,random_state=seed)
        return r1 + r2


class BVN():
    def __init__(self, mu, sigma, rho):
        """
        mu: array of means
        sigma: array of variances
        rho: correlation coefficient
        """
        if isinstance(mu,list):
            mu, sigma = np.array(mu), np.array(sigma)
        assert mu.shape[0]==sigma.shape[0]==2
        assert np.abs(rho) <= 1
        self.mu = mu.reshape([1,2])
        self.sigma = sigma.flatten()
        od = rho*np.sqrt(sigma.prod())
        self.rho = rho
        self.Sigma = np.array([[sigma[0],od],[od, sigma[1]]])
        self.A = cholesky(self.Sigma) # A.T.dot(A) = Sigma

    # size=1000;seed=1234 # del size, seed
    def rvs(self, size, seed=None):
        """
        size: number of samples to simulate
        seed: to pass onto np.random.seed
        """
        np.random.seed(seed)
        X = np.random.randn(size,2)
        Z = self.A.T.dot(X.T).T + self.mu
        return Z

    def imills(self, a):
        return norm.pdf(a)/norm.cdf(-a)

    def sheppard(self, theta, h, k):
        return (1/(2*np.pi))*np.exp(-0.5*(h**2+k**2-2*h*k*np.cos(theta))/(np.sin(theta)**2))

    # h, k = -2, -np.infty
    def orthant(self, h, k, method='scipy'):
        # P(X1 >= h, X2 >=k)
        assert method in ['scipy','cox','sheppard']
        if isinstance(h,int) or isinstance(h, float):
            h, k = np.array([h]), np.array([k])
        else:
            assert isinstance(h,np.ndarray) and isinstance(k,np.ndarray)
        assert len(h) == len(k)
        # assert np.all(h >= 0) and np.all(k >= 0)
        # Calculate the number of standard deviations away it is        
        Y = (np.c_[h, k] - self.mu)/np.sqrt(self.sigma)
        Y1, Y2 = Y[:,0], Y[:,1]
        
        # (i) scipy: L(h, k)=1-(F1(h)+F2(k))+F12(h, k)
        if method == 'scipy':
            sp_bvn = MVN([0, 0],[[1,self.rho],[self.rho,1]])
            pval = 1+sp_bvn.cdf(Y)-(norm.cdf(Y1)+norm.cdf(Y2))
            return pval 

        # A Simple Approximation for Bivariate and Trivariate Normal Integrals
        if method == 'cox':
            mu_a = self.imills(Y1)
            root = np.sqrt(1-self.rho**2)
            xi = (self.rho * mu_a - Y2) / root
            pval = norm.cdf(-Y1) * norm.cdf(xi)
            return pval

        if method == 'sheppard':
            pval = np.array([quad(self.sheppard, np.arccos(self.rho), np.pi, args=(y1,y2))[0] for y1, y2 in zip(Y1,Y2)])
            return pval

# dist_BVN = BVN(mu=np.array([1,2]),sigma=np.array([2,3]), rho=0.5)
# self = dist_BVN