import os
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, chi2, t
from plotnine import *
from timeit import timeit
from time import time

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from classes import BVN, NTS, two_stage, ols, dgp_yX, tnorm

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'figures')

#############################
# --- (4C) DATA CARVING --- #

# Truncated normal specification
alpha = 0.05
dist = tnorm(mu=0, sig2=1, a=1, b=np.inf)
np.round(pd.DataFrame({'CI_2.5%':dist.CI(dist.ppf(alpha/2)),'CI_97.%':dist.CI(dist.ppf(1-alpha/2))}),1)

cutoff = 0.1
n, p, sig2 = 100, 20, 1
m = int(n/2)
beta_null = np.repeat(0, p)
nsim = 2000
holder_ols, holder_SI = [], []
for i in range(nsim):
    if (i+1) % 100 == 0:
        print(i+1)
    resp, xx = dgp_yX(n=n, p=p, snr=1, b0=0, seed=i)
    # Data splitting
    mdl1 = ols(y=resp[:m], X=xx[:m], has_int=False, sig2=sig2)
    abhat1 = np.abs(mdl1.bhat)
    M1 = abhat1 > cutoff
    if M1.sum() > 0:
        mdl2 = ols(y=resp[m:], X=xx[m:,M1], has_int=False, sig2=sig2)
        mdl2.CI(alpha)
        tmp_ols = pd.DataFrame({'sim':i,'bhat':mdl2.bhat,'lb':mdl2.lb,'ub':mdl2.ub,'z':mdl2.z})
        holder_ols.append(tmp_ols)
    # Selective inference
    mdl = ols(y=resp, X=xx, has_int=False, sig2=sig2)
    M = np.abs(mdl.bhat)>cutoff
    if M.sum() > 0:        
        tmp_M = pd.DataFrame({'sim':i,'bhat':mdl.bhat[M],'Vjj':np.diagonal(mdl.covar)[M]})
    holder_SI.append(tmp_M)
# Is truncated Gaussian?
dat_TN = pd.concat(holder_SI).assign(tt='TN')
dat_TN = dat_TN.assign(abhat=lambda x: x.bhat.abs(), sbhat=lambda x: np.sign(x.bhat).astype(int))
dist_TN_ols = tnorm(mu=0,sig2=dat_TN.Vjj,a=cutoff,b=np.inf)
pval_TN = dist_TN_ols.cdf(dat_TN.abhat)
dat_TN['pval'] = 2*np.minimum(pval_TN,1-pval_TN)
dat_TN = dat_TN.sort_values('abhat',ascending=False).reset_index(None,True)

# Speed up solution with warm start
lb, ub = 0.2, 0.7
stime, holder = time(), []
for j, (ab, vj) in enumerate(zip(dat_TN.abhat, dat_TN.Vjj)):
    if j < 14000:
        continue
    if j % 25 == 0:
        nleft, rate = dat_TN.shape[0] - (j+1), (j+1)/(time() - stime)
        seta = nleft/rate
        print('ETA: %0.1f seconds (%i, %i)' % (seta, j, dat_TN.shape[0]))
    tmp_dist = tnorm(mu=0,sig2=vj, a=cutoff, b=np.inf)
    if j == 1:
        lb, ub = tmp_dist.CI(x=ab,alpha=alpha,method='BFGS',lb=0.2,ub=0.7)
    else:
        lb, ub = tmp_dist.CI(x=ab,alpha=alpha,method='BFGS',lb=lb,ub=ub)
    lb, ub = lb[0], ub[0]
    if (np.sign(lb) != np.sign(ub)) & (dat_TN.pval[j] < alpha):
        print('woops!!')
        break
    tmp = pd.DataFrame({'lb':lb,'ub':ub},index=[j])
    holder.append(tmp)

# dat_CI = pd.concat(holder).reset_index(None,True)

dat_CI.iloc[50:60]

# Add on the UB/LB
dat_TN = pd.concat([dat_TN, dat_CI],1)
dat_TN = dat_TN.assign(reject=lambda x: x.pval < alpha, cover=lambda x: np.sign(x.lb)!=np.sign(x.ub))

dat_TN.query('reject==True & cover==True').index
dat_TN.groupby(['reject','cover']).size()

dat_TN[['reject','cover']].mean()


# Check for coverage
dat_ols = pd.concat(holder_ols).assign(tt='LS')

dat_sim = pd.concat([dat_ols, dat_TN]).reset_index(None,True)
dat_sim = dat_sim.assign(cover=lambda x: np.sign(x.lb)!=np.sign(x.ub), reject=lambda x: x.pval<alpha)
dat_sim.groupby('tt').apply(lambda x: pd.Series({'pval':x.reject.mean(),'cover':x.cover.mean()}))
dat_sim.groupby(['tt','cover','reject']).size()
dat_sim.query('tt=="TN"').groupby(['cover','reject']).size()

np.quantile(np.abs(dat_TN.bhat),0.025)
tnorm(mu=0,sig2=0.0107,a=cutoff,b=np.inf).ppf(0.025)
np.quantile(np.abs(dat_TN.bhat),0.975)
tnorm(mu=0,sig2=0.0107,a=cutoff,b=np.inf).ppf(0.975)




#############################
# --- (4A) QUERIES 1964 --- #
# X~N(100,6), Y~TN(50,3,44,Inf)

mu1, tau1 = 100, 6
mu2, tau2, a, b = 50, 3, 44, np.inf
mu, tau = np.array([mu1, mu2]), np.array([tau1,tau2])
dist_A = NTS(mu=mu,tau=tau, a=a, b=b)
dist_A.cdf(138)


#######################################
# --- (2) BVN INTEGRATION METHODS --- #

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

####################################
# --- (3) NORMAL-TRUNCATED-SUM --- #

# Demonstrated with example
mu1, tau1 = 1, 1
mu2, tau2, a, b = 1, 2, -1, 4
mu, tau = np.array([mu1, mu2]), np.array([tau1,tau2])
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


#####################################
# --- (4B) TWO-STAGE REGRESSION --- #

delta, sigma2 = 2, 4
n, m = 100, 100
gamma, alpha = 0.01, 0.05
nsim = 10000000
p_seq = np.arange(0.01,1,0.01)
two_stage(n=n, m=m, gamma=gamma, alpha=alpha, pool=True).power[0]

# --- (A) CALCULATE N=M=100 PP-QQ PLOT --- #
dist_2s = two_stage(n=n, m=m, gamma=gamma, alpha=alpha, pool=True)
print('Power: %0.3f' % dist_2s.power)
df_2s = dist_2s.rvs(nsim=nsim, delta=delta, sigma2=sigma2)
df_2s = df_2s.assign(Null=lambda x: x.d0hat < delta)
df_2s = df_2s.assign(reject=lambda x: x.shat < dist_2s.t_alpha)
df_2s.groupby('Null').reject.mean()

qq_emp = df_2s.groupby('Null').apply(lambda x: pd.DataFrame({'pp':p_seq,'qq':np.quantile(x.shat,p_seq)}))
qq_emp = qq_emp.reset_index().drop(columns='level_1')
qq_theory_H0 = dist_2s.H0.ppf(p_seq)
qq_theory_HA = dist_2s.HA.ppf(p_seq)
tmp1 = pd.DataFrame({'pp':p_seq,'theory':qq_theory_H0,'Null':True})
tmp2 = pd.DataFrame({'pp':p_seq,'theory':qq_theory_HA,'Null':False})
qq_pp = qq_emp.merge(pd.concat([tmp1, tmp2]))
qq_pp = qq_pp.melt(['pp','Null'],['qq','theory'],'tt')

gtit = 'n=%i, m=%i, nsim=%i' % (n, m, nsim)
gg_qp_2s = (ggplot(qq_pp,aes(x='pp',y='value',color='tt')) + 
    theme_bw() + geom_line() +  ggtitle(gtit) + 
    facet_wrap('~Null',scales='free_y',labeller=label_both,ncol=1) + 
    labs(x='Percentile',y='Quantile') + 
    theme(legend_position=(0.3,0.8)) + 
    scale_color_discrete(name='Method',labels=['Empirical','Theory']))
gg_qp_2s.save(os.path.join(dir_figures,'gg_qp_2s.png'),width=4,height=7)

# --- (B) POWER AS GAMMA VARIES --- #
gamma_seq = np.round(np.arange(0.01,0.21,0.01),2)
power_theory = np.array([two_stage(n=n, m=m, gamma=g, alpha=alpha, pool=False).power[0] for g in gamma_seq])
ub_theory = delta + np.sqrt(sigma2/n)*t(df=n-1).ppf(1-gamma_seq)
power_emp, ub_emp = np.zeros(power_theory.shape), np.zeros(ub_theory.shape)
for i, g in enumerate(gamma_seq):
    print('%i of %i' % (i+1, len(gamma_seq)))
    tmp_dist = two_stage(n=n, m=m, gamma=g, alpha=alpha, pool=False)
    tmp_sim = tmp_dist.rvs(nsim=nsim, delta=delta, sigma2=sigma2)
    tmp_sim = tmp_sim.assign(Null=lambda x: x.d0hat < delta, 
                reject=lambda x: x.shat < tmp_dist.t_alpha)
    power_emp[i] = tmp_sim.query('Null==False').reject.mean()
    ub_emp[i] = tmp_sim.d0hat.mean()

tmp1 = pd.DataFrame({'tt':'theory','gamma':gamma_seq,'power':power_theory,'ub':ub_theory})
tmp2 = pd.DataFrame({'tt':'emp','gamma':gamma_seq,'power':power_emp,'ub':ub_emp})
dat_gamma = pd.concat([tmp1, tmp2]).melt(['tt','gamma'],None,'msr')
di_msr = {'power':'Power','ub':'delta0'}

gg_gamma = (ggplot(dat_gamma,aes(x='gamma',y='value',color='tt')) + 
    theme_bw() + geom_line() +  ggtitle(gtit) + 
    facet_wrap('~msr',scales='free_y',ncol=1,labeller=labeller(msr=di_msr)) + 
    labs(x='gamma',y='Value') + 
    theme(legend_position=(0.6,0.8)) + 
    scale_x_continuous(limits=[0,0.2]) + 
    scale_color_discrete(name='Method',labels=['Empirical','Theory']))
gg_gamma.save(os.path.join(dir_figures,'gg_gamma.png'),width=4,height=7)

# --- (C) POWER AS N = K - M VARIES --- #

k = n + m
n_seq = np.arange(5,k,5)
dat_nm = pd.concat([pd.DataFrame({'n':nn,'m':k-nn,
    'power':two_stage(n=nn, m=k-nn, gamma=gamma, alpha=alpha, pool=True).power[0]},index=[nn]) for nn in n_seq])
dat_nm = dat_nm.reset_index(None,True).assign(ub=delta + np.sqrt(sigma2/n_seq)*t(df=n_seq-1).ppf(1-gamma))
dat_nm = dat_nm.melt(['n','m'],None,'msr')

gg_nm = (ggplot(dat_nm,aes(x='n',y='value')) + 
    theme_bw() + geom_line() +  
    ggtitle('gamma=0.01, m=200-n') + 
    facet_wrap('~msr',scales='free_y',ncol=1,labeller=labeller(msr=di_msr)) + 
    labs(x='n',y='Value'))
gg_nm.save(os.path.join(dir_figures,'gg_nm.png'),width=4,height=7)



