import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from plotnine import *
from timeit import timeit

from classes import BVN, NTS

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'figures')

###########################################
# --- COMPARE BVN INTEGRATION METHODS --- #

# https://stats.stackexchange.com/questions/61080/how-can-i-calculate-int-infty-infty-phi-left-fracw-ab-right-phiw?noredirect=1&lq=1
# https://math.stackexchange.com/questions/449875/expected-value-of-normal-cdf/1125935
# https://math.stackexchange.com/questions/2392827/normal-random-variable-as-argument-of-standard-normal-cdf


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


################################
# --- TWO-STAGE REGRESSION --- #

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from plotnine import *
from timeit import timeit

from classes import BVN, NTS

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'figures')

# http://www.erikdrysdale.com/figures/power_for_regression_metrics_4_0.png
# GRID OF N1/N2
# WHATS HAPPENING TO THE CRITICAL VALUE? 
# WHATS HAPPENING TO THE MEAN OF S0, SA?
# WHAT'S HAPPENING TO DELTA | H0, HA?



# WRITE A FUNCTION WRAPPER TO GENERATE DATA


delta, sigma2 = 2, 4
n, m = 1749, 1
gamma, alpha = 0.1, 0.05

# DISTRIBUTIONS 
mu_2stage = np.array([0, -np.sqrt(m/n)*norm.ppf(1-gamma)])
tau_2stage = np.sqrt([1, m/n])
dist_2s_H0 = NTS(mu=mu_2stage,tau=tau_2stage, a=0, b=np.infty)
dist_2s_HA = NTS(mu=mu_2stage,tau=tau_2stage, a=-np.infty, b=0)
crit_val = dist_2s_H0.ppf(alpha)[0]
power = 1 - dist_2s_HA.cdf(crit_val)
print('Power: %0.3f' % power)


# SIMULATIONS

# Compare to simulation
nsim = 50000
np.random.seed(nsim)
S = delta+np.sqrt(sigma2)*np.random.randn(nsim,n)
T = delta+np.sqrt(sigma2)*np.random.randn(nsim,m)
delta1, delta2 = S.mean(1), T.mean(1)
# ESTIMATE 

sigS, sigT = S.std(1,ddof=1), T.std(1,ddof=1)
del S, T
delta0 = delta1 + (sigS/np.sqrt(n))*norm.ppf(1-gamma)
shat = (delta2 - delta0)/(sigT/np.sqrt(m))
p_seq = np.arange(0.01,1,0.01)
s_null_H0 = shat[delta > delta0]
s_null_HA = shat[delta < delta0]
qq_emp_H0 = np.quantile(s_null_H0,p_seq)
qq_emp_HA = np.quantile(s_null_HA,p_seq)
qq_theory_H0 = dist_2s_H0.ppf(p_seq)
qq_theory_HA = dist_2s_H0.ppf(p_seq)
tmp1 = pd.DataFrame({'pp':p_seq,'emp':qq_emp_H0,'theory':qq_emp_H0,'Null':'H0'})
tmp2 = pd.DataFrame({'pp':p_seq,'emp':qq_emp_HA,'theory':qq_emp_HA,'Null':'HA'})
dat_2stage = pd.concat([tmp1, tmp2]).melt(['pp','Null'],None,'tt','qq')
# dat_2stage.pivot_table('qq',['pp','Null'],'tt').reset_index().groupby('Null').apply(lambda x: np.corrcoef(x.emp,x.theory)[0,1])

gg_2stage_alpha = (ggplot(dat_2stage,aes(x='pp',y='qq',color='tt')) + 
    theme_bw() + geom_line() +  
    facet_wrap('~Null',scales='free',labeller=label_both) + 
    labs(x='Percentile',y='Quantile') + 
    theme(subplots_adjust={'wspace': 0.15}) + 
    scale_color_discrete(name='Method',labels=['Empirical','Theory']))
gg_2stage_alpha.save(os.path.join(dir_figures,'gg_2stage_alpha.png'),width=7,height=3.5)

# Compare predicted to actual power (SEQUENCE OVER TAU)
n_seq = np.arange(50,750,50)
nm = 750
m_seq = nm - n_seq

holder = []
np.random.seed(nm)
for n in n_seq:
    m = nm - n
    print('n: %i, m: %i' % (n, m))
    mu_2stage = np.array([0, -np.sqrt((m+n)/n)*norm.ppf(1-gamma)])
    tau_2stage = np.sqrt([1, (m+n)/n])
    dist_2s_H0 = NTS(mu=mu_2stage,tau=tau_2stage, a=0, b=np.infty)
    dist_2s_HA = NTS(mu=mu_2stage,tau=tau_2stage, a=-np.infty, b=0)
    crit_val_theory = dist_2s_H0.ppf(alpha)[0]
    power_theory = 1 - dist_2s_HA.cdf(crit_val_theory)
    power_theory
    # Compare to reality
    S, T = np.random.randn(nsim,n), np.random.randn(nsim,m)
    delta1, delta2 = S.mean(1), T.mean(1)
    sigS, sigT = S.std(1,ddof=1), np.c_[S,T].std(1,ddof=1)
    del S, T
    delta0 = delta1 + (sigS/np.sqrt(n))*norm.ppf(1-gamma)
    shat = (delta2 - delta0)/(sigT/np.sqrt(n+m))
    s_null_H0 = shat[delta0 < 0]  # Null true
    s_null_HA = shat[delta0 > 0]  # Null false
    crit_val_emp = np.quantile(s_null_H0,alpha)    
    power_emp = np.mean(s_null_HA > crit_val_emp)
    # Store: 'mu':mu_2stage[1],'tau':tau_2stage[1]
    power_emp
    tmp = pd.DataFrame({'n':n, 'm':m, 'gamma':gamma,
                    'mu_W':dist_2s_HA.mu_W,'delta_hat':delta0.mean(),
                    'theory':power_theory,'emp':power_emp},index=[0])
    holder.append(tmp)
# Merge and analyze
dat_power_2s = pd.concat(holder).reset_index(None,True)
dat_power_2s[['n','m','theory','emp']]


# Compare along a sequence of gamma's
gamma_seq = np.round(np.arange(0.01,0.21,0.01),2)






########################
# --- QUERIES 1964 --- #
# X~N(100,6), Y~TN(50,3,44,Inf)

