"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

# External
import numpy as np
from scipy.stats import beta, betabinom
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR
from mapie.classification import MapieClassifier  # Look at line 1207
# https://mapie.readthedocs.io/en/stable/theoretical_description_classification.html#adaptive-prediction-sets-aps
# https://mapie.readthedocs.io/en/stable/generated/mapie.classification.MapieClassifier.html
# Internal
from _rmd.extra_conformal.utils import dgp_multinomial, \
                                        dgp_continuous, \
                                        NoisyGLM, simulation_cp, \
                                        LinearQuantileRegressor, \
                                        QuantileRegressors
from _rmd.extra_conformal.conformal import conformal_sets, \
                                            score_lac, score_aps, \
                                            score_mae, score_mse, \
                                                score_pinpall
# Ignore warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


##########################
# --- (1) SIM PARAMS --- #

# Set parameters
seed = 12
nsim = 500
p = 3
k = 4
snr_k = 0.5 * k
# Specify data sizes
n_train = 250
n_calib = 500
n_val = 100
# Error rate
alpha = 0.1
# Expected distribution of coverage
r = np.floor((n_calib + 1) * alpha)
a = n_calib + 1 - r
b = r
dist_cover_cond = beta(a=a, b=b)
dist_cover_marg = betabinom(n=n_val, a=a, b=b)

print('~~~~~~~~~~~~~~~')
print(f'Theory: cover lb = {(1-alpha):.3f}, ub={1-alpha+1/(n_calib+1):.3f}')

run_class = True
run_reg = False

sim_kwargs = {'n_train':n_train, 'n_calib':n_calib, 'n_test': n_val, 
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
    conformalizer = conformal_sets(f_theta=mdl, score_fun=score_lac, alpha=alpha, upper=True)
    # Compare coverage
    simulator = simulation_cp(dgp=data_generating_process, 
                              ml_mdl=mdl, 
                              cp_mdl=conformalizer, 
                              is_classification=True,)
    res_class = simulator.run_simulation(**sim_kwargs, force_redraw=True, n_iter=250, verbose=True)
    print('~~~ Classification sets ~~~')
    print(f"CP: coverage={100*res_class['cover'].mean():.1f}%, set size={res_class['set_size'].mean():.2f}, q={res_class['qhat'].mean():.3f}")
    pval = dist_cover_marg.cdf(np.mean(n_val * res_class['cover']))
    pval = np.minimum(pval, 1-pval) * 2
    print(f"P-value for observered coverage = {100*pval:.1f}%")
    print('\n')


##########################
# --- (2) REGRESSION --- #

if run_reg:
    data_generating_process = dgp_continuous(p, k, snr=snr_k, seeder=seed)
    # (i) Standard method
    mdl = NoisyGLM(noise_std = 0.0, seeder=seed, subestimator=LinearRegression)
    conformalizer = conformal_sets(f_theta=mdl, score_fun=score_mse, alpha=alpha, upper=True)
    
    # # (ii) Quantile method (for GBR: {..., n_estimators=50, loss='quantile'})
    # mdl = QuantileRegressors(noise_std = 0.0, seeder=seed, subestimator=LinearQuantileRegressor, alphas=[alpha/2, 1-alpha/2])
    # conformalizer = conformal_sets(f_theta=mdl, score_fun=score_pinpall, alpha=alpha, upper=True)
    
    # (iii) Simulation
    simulator = simulation_cp(dgp=data_generating_process, 
                              ml_mdl=mdl, 
                              cp_mdl=conformalizer, 
                              is_classification=False)
    res_reg = simulator.run_simulation(**sim_kwargs, verbose=True, n_iter=50)
    print('~~~ Regression intervals ~~~')
    print(f"CP: coverage={100*res_reg['cover'].mean():.1f}%, interval width={res_reg['set_size'].mean():.2f}, q={res_reg['qhat'].mean():.3f}")
    pval = dist_cover_marg.cdf(np.mean(n_val * res_reg['cover']))
    pval = np.minimum(pval, 1-pval) * 2
    print(f"P-value for observered coverage = {100*pval:.1f}%")
    print('\n')


