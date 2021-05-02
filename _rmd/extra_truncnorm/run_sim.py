import os
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.stats import multivariate_normal as MVN
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from plotnine import *
from timeit import timeit

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

    # self=dist_BVN; h, k = np.array([1,2,3]), np.array([2,3,4])
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
            pval = [quad(self.sheppard, np.arccos(self.rho), np.pi, args=(y1,y2))[0] for y1, y2 in zip(Y1,Y2)]
            return pval


# Set up a BVN
mu = np.array([1,2])
sigma = np.array([0.5,2])
dist_BVN = BVN(mu,sigma,rho=0.4)
nrun = 10000

# Check that different approaches line up
h_seq = np.linspace(mu[0],mu[0]+2*sigma[0],5)
k_seq = np.linspace(mu[1],mu[1]+2*sigma[1],5)
n_std = np.arange(len(h_seq))/2
methods = ['scipy','cox','sheppard']

holder = []
for h in h_seq:
    for k in k_seq:
        print('h: %0.1f, k: %0.1f' % (h,k))
        ptime = np.zeros([len(methods),2])
        for i, method in enumerate(methods):
            tmp_pv = dist_BVN.orthant(h, k, method)
            tmp_time = timeit('dist_BVN.orthant(h, k, method)',number=nrun,globals=globals())
            ptime[i] = [tmp_pv, tmp_time]
        tmp_df = pd.DataFrame(ptime,columns=['pval','rtime']).assign(h=h,k=k,method=methods)
        holder.append(tmp_df)
df_pval = pd.concat(holder).reset_index(None,True)
df_pval = df_pval.assign(h=lambda x: (x.h-mu[0])/sigma[0], k=lambda x: (x.k-mu[1])/sigma[1])
df_pval = df_pval.assign(xlbl=lambda x: '('+x.h.astype(str)+','+x.k.astype(str)+')')

# Repeat timing for vectorized operations
nrun = 1000
ll = range(50,501,50)
holder = []
methods2 = ['cox','scipy']
for l in ll:
    print(l)
    h = np.arange(l) / l
    k = h + 1
    tmp = np.repeat(0.0,len(methods2))
    for i,method in enumerate(methods2):
        tmp[i] = timeit('dist_BVN.orthant(h, k, method)',number=nrun,globals=globals())
    holder.append(pd.DataFrame({'method':methods2,'rtime':tmp,'ll':l}))
df_vec = pd.concat(holder).reset_index()

posd = position_dodge(0.5)
# Vectorized run times
gg_rvec = (ggplot(df_vec,aes(x='ll',y='rtime',color='method')) + 
    theme_bw() + geom_line() + geom_point() + 
    labs(x='Vector length',y='Run time for 1K calls') + 
    scale_color_manual(name='CDF method',values=["#F8766D","#00BA38"]) + 
    scale_x_continuous(breaks=list(np.array(ll))) + 
    guides(color=False))
gg_rvec.save(os.path.join(dir_figures,'gg_rvec.png'),width=4,height=4.5)

# P-values
gg_pval = (ggplot(df_pval,aes(x='xlbl',y='-np.log2(pval)',color='method')) + 
    theme_bw() + scale_color_discrete(name='CDF method') + 
    geom_point(size=1,position=posd) + 
    labs(x='(h,k) std dev',y='-log2 orthant probability') + 
    theme(axis_text_x=element_text(angle=90)))
gg_pval.save(os.path.join(dir_figures,'gg_pval.png'),width=7,height=4.5)

# Run times
gg_rtime = (ggplot(df_pval,aes(x='xlbl',y='rtime',color='method')) + 
    theme_bw() + scale_color_discrete(name='CDF method') +  
    geom_point(size=1,position=posd) + 
    labs(x='(h,k) std dev',y='Run time for 10K calls') + 
    theme(axis_text_x=element_text(angle=90)))
gg_rtime.save(os.path.join(dir_figures,'gg_rtime.png'),width=7,height=4.5)

################################
# --- NORMAL-TRUNCATED-SUM --- #

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
        orthant1 = bvn.orthant(m1,np.repeat(self.alpha,nx),method=method)
        orthant2 = bvn.orthant(m1,np.repeat(self.beta,nx),method=method)
        return 1 - (orthant1 - orthant2)/self.Z

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


# Demonstrated with example
mu1, tau1 = 1, 1
mu2, tau2, a, b = 1, 2, -1, 4
mu, tau = np.array([mu1, mu2]), np.array([tau1,tau2])**2
dist_NTS = NTS(mu=mu,tau=tau, a=a, b=b)
n_iter = 100000
W_sim = dist_NTS.rvs(n=n_iter,seed=1)
mu_sim, mu_theory = W_sim.mean(),dist_NTS.mu_W
xx = np.linspace(-5*mu.sum(),5*mu.sum(),n_iter)
mu_quad = np.sum(xx*dist_NTS.pdf(xx)*(xx[1]-xx[0]))
methods = ['Empirical','Theory', 'Quadrature']
mus = [mu_sim, mu_theory, mu_quad]
# Compare
print(pd.DataFrame({'method':methods,'mu':mus}))

# Do quadrature approximation
dat_W_sim = pd.DataFrame({'W':W_sim,'f':dist_NTS.pdf(W_sim)})

# Plot empirical frequency to theory
gg_Wsim = (ggplot(dat_W_sim) + 
    theme_bw() +
    labs(x='W',y='Simulation density') + 
    ggtitle('Blue curve shows Eq. (3)') + 
    stat_bin(aes(x='W',y=after_stat('density')),bins=40,geom='geom_histogram',
        fill='red',alpha=0.5,color='black') +  
    stat_function(aes(x='W'),fun=dist_NTS.pdf,color='blue'))
gg_Wsim.save(os.path.join(dir_figures,'gg_Wsim.png'),width=5,height=4)

# Create a P-P plot
p_seq = np.arange(0.005,1,0.005)
q_seq = np.quantile(W_sim,p_seq)
p_theory = dist_NTS.cdf(q_seq)
q_theory = dist_NTS.ppf(p_seq)

tmp1 = pd.DataFrame({'tt':'Percentile','theory':p_theory,'emp':p_seq}).reset_index()
tmp2 = pd.DataFrame({'tt':'Quantile','theory':q_theory,'emp':q_seq}).reset_index()
dat_ppqq = pd.concat([tmp1,tmp2]).reset_index(None, True)

# Plot empirical frequency to theory
gg_ppqq = (ggplot(dat_ppqq,aes(x='theory',y='emp')) + 
    theme_bw() + geom_point(size=0.5) + 
    geom_abline(slope=1,intercept=0,linetype='--',color='blue') + 
    facet_wrap('~tt',scales='free') + 
    labs(x='Theory',y='Empirical') + 
    theme(subplots_adjust={'wspace': 0.15}))
gg_ppqq.save(os.path.join(dir_figures,'gg_ppqq.png'),width=8,height=3.5)



########################
# --- QUERIES 1964 --- #
# X~N(100,6), Y~TN(50,3,44,Inf)





    
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