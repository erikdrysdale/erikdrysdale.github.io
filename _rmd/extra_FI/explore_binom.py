import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from scipy.stats import norm
from funs_explore import BPFI, gg_save

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
assert os.path.exists(dir_figures), 'no figures folder found'

######################################
# ---- (1) CHECK FI CALCULATION ---- #

n = 100000
pi2 = 0.6
pi1 = 0.5
alpha = 0.05

sim1 = BPFI(n=n, pi1=pi1, pi2=pi2, alpha=alpha)
sim1.dgp(nsamp=10000)
sim1.fi()
sim1.mu_f
sim1.df_fi.query('reject==1').mean()


gtit='Negative values imply reverse FI\n$\\pi_d=0.1,\\pi_1=0.5,n=100$'
gg_fi_approx = (pn.ggplot(sim1.df_fi,pn.aes(x='fia',y='fi')) + 
    pn.theme_bw() + pn.labs(x='BPFI-approx',y='BPFI-exact') + 
    pn.ggtitle(gtit) + pn.geom_point() + 
    pn.geom_abline(intercept=0,slope=1,color='blue',linetype='--'))
gg_save('gg_fi_approx.png',dir_figures,gg_fi_approx,6,4)





###########################################
# ---- (1) ESTIMATE QUALITY OF POWER ---- #

alpha = 0.05
nsim = 1000000
n_seq = [25, 100, 500, 2000]
pi1_seq = [0.05, 0.25, 0.5]
pid_seq = [0, 0.01, 0.05, 0.1]
n_perm = len(n_seq) * len(pi1_seq) * len(pid_seq)

holder = []
jj = 0
for n in n_seq:
    for pi1 in pi1_seq:
        for pid in pid_seq:
            jj += 1
            print('Iteration %i of %i' % (jj, n_perm))
            pi2 = pi1 + pid
            tmp_sim = binom_experiment(n=n, pi1=pi1, pi2=pi2, alpha=alpha)
            tmp_sim.dgp(nsamp=nsim, seed=jj)
            power_norm = tmp_sim.reject.mean()
            power_t = np.mean(1 - stats.t(df=2*n).cdf(tmp_sim.z) < alpha)
            tmp_slice = pd.DataFrame({'n':n, 'pi1':pi1, 'pid':pid, 'power_est':tmp_sim.power, 'power_norm':power_norm, 'power_t':power_t}, index=[jj])
            holder.append(tmp_slice)
# Merge and plot            
res_power = pd.concat(holder)
res_power







