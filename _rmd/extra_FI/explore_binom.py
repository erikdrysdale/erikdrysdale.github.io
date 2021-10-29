import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from funs_explore import binom_experiment, gg_save

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
assert os.path.exists(dir_figures), 'no figures folder found'

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







