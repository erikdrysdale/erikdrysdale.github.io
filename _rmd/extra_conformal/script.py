"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

# External
import numpy as np
from scipy.stats import beta, binom, betabinom
from sklearn.linear_model import LogisticRegression, LinearRegression, QuantileRegressor
from mapie.classification import MapieClassifier  # Look at line 1207
# Internal
from _rmd.extra_conformal.utils import dgp_multinomial, dgp_continuous, \
                            NoisyGLM, simulation_cp
from _rmd.extra_conformal.conformal import conformal_sets, \
                                            score_ps, score_aps, \
                                            score_mae, score_mse
# Ignore warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


##########################
# --- (1) SIM PARAMS --- #

# Set parameters
seed = 123
nsim = 500
p = 3
k = 4
snr_k = 0.5 * k
# Specify data sizes
n_train = 250
n_calib = 100
n_eval = 1
# Error rate
alpha = 0.1
# Expected distribution of coverage
r = np.floor((n_calib + 1) * alpha)
a = n_calib + 1 - r
b = r
dist_cover_cond = beta(a=a, b=b)
dist_cover_marg = betabinom(n=n_eval*nsim, a=a, b=b)

print('~~~~~~~~~~~~~~~')
print(f'Theory: cover lb = {(1-alpha):.3f}, ub={1-alpha+1/(n_calib+1):.3f}')

run_class = False
run_reg = True
run_quant = False

sim_kwargs = {'n_train':n_train, 'n_calib':n_calib, 'n_test': n_eval, 
              'nsim':nsim, 'seeder':seed, }


##############################
# --- (1) CLASSIFICATION --- #

if run_class:
    # Set up data generating process (DGP)
    data_generating_process = dgp_multinomial(p, k, snr=snr_k, seeder=seed)
    # Pick model class
    mdl = NoisyGLM(max_iter=250, noise_std = 0.0, seeder=seed, 
                                subestimator=LogisticRegression,
                                penalty=None, )
    # Set up conformalizer
    conformalizer = conformal_sets(f_theta=mdl, score_fun=score_ps, alpha=alpha, upper=True)
    # Compare coverage
    simulator = simulation_cp(dgp=data_generating_process, 
                              ml_mdl=mdl, 
                              cp_mdl=conformalizer, 
                              is_classification=True)
    res_class = simulator.run_simulation(**sim_kwargs, force_redraw=True)
    print('~~~ Classification sets ~~~')
    print(f"CP: coverage={100*res_class['cover'].mean():.1f}%, set size={res_class['set_size'].mean():.2f}, q={res_class['qhat'].mean():.3f}")
    pval = dist_cover_marg.cdf(res_class['cover'].sum())
    pval = np.minimum(pval, 1-pval) * 2
    print(f"P-value for observered coverage = {100*pval:.1f}%")
    print('\n')


##########################
# --- (2) REGRESSION --- #

if run_reg:
    data_generating_process = dgp_continuous(p, k, snr=snr_k, seeder=seed)
    mdl = NoisyGLM(noise_std = 0.0, seeder=seed, subestimator=LinearRegression)
    conformalizer = conformal_sets(f_theta=mdl, score_fun=score_mae, alpha=alpha, upper=True)
    simulator = simulation_cp(dgp=data_generating_process, 
                              ml_mdl=mdl, 
                              cp_mdl=conformalizer, 
                              is_classification=False)
    res_reg = simulator.run_simulation(**sim_kwargs)
    print('~~~ Regression intervals ~~~')
    print(f"CP: coverage={100*res_reg['cover'].mean():.1f}%, set size={res_reg['set_size'].mean():.2f}, q={res_reg['qhat'].mean():.3f}")
    pval = dist_cover_marg.cdf(res_reg['cover'].sum())
    pval = np.minimum(pval, 1-pval) * 2
    print(f"P-value for observered coverage = {100*pval:.1f}%")
    print('\n')


