"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

import numpy as np
import pandas as pd
from scipy.stats import beta, binom
from sklearn.linear_model import LogisticRegression
# Internal
from _rmd.extra_conformal.utils import dgp_multinomial, NoisyLogisticRegression, simulation_classification
from _rmd.extra_conformal.conformal import classification_sets, score_ps

##########################
# --- (0) SIM PARAMS --- #

# Set parameters
seed = 123
nsim = 500
p = 3
k = 4
snr_k = 0.5 * k


######################
# --- (1) SET UP --- #

# Specify data sizes
n_train = 250
n_calib = 100

# Error rate
alpha = 0.1

# Expected distribution of coverage
dist_cover_marg = binom(n=nsim, p=1-alpha)
r = np.floor((n_calib + 1) * alpha)
dist_cover_cond = beta(a=n_calib + 1 - r, b=r)

# Set up data generating process (DGP)
data_generating_process = dgp_multinomial(p, k, snr=snr_k, seeder=seed)

# Pick model class
mdl = NoisyLogisticRegression(max_iter=250, noise_std = 0.0, seeder=seed, 
                              subestimator=LogisticRegression,
                              penalty=None, )
# Set up conformalizer
conformalizer = classification_sets(f_theta=mdl, score_fun=score_ps, alpha=alpha, upper=True)


##########################
# --- (2) SIMULATION --- #

print('~~~~~~~~~~~~~~~')
print(f'Theory: cover lb = {(1-alpha):.3f}, ub={1-alpha+1/(n_calib+1):.3f}')
# Compare coverage
res_simul = simulation_classification(dgp=data_generating_process, 
                           ml_mdl=mdl, 
                           cp_mdl=conformalizer, 
                           n_train=n_train, n_calib=n_calib,
                           nsim=nsim, seeder=seed)
print(f"CP: coverage={100*res_simul['cover'].mean():.1f}%, set size={res_simul['set_size'].mean():.2f}, q={res_simul['qhat'].mean():.3f}")
pval = dist_cover_marg.cdf(res_simul['cover'].sum())
pval = np.minimum(pval, 1-pval) * 2
print(f"Prob of obsering coverage = {100*pval:.0f}%")
print('\n')