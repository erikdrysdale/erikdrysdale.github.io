"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

# Path
import os
dir_base = os.getcwd()
dir_figs = '_rmd/extra_conformal'

# External
import numpy as np
import pandas as pd
import plotnine as pn
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
r = n_calib - np.ceil((n_calib + 1) * (1-alpha))
a = n_calib + 1 - r
b = r
dist_cover_cond = beta(a=a, b=b)
dist_cover_marg = betabinom(n=n_val, a=a, b=b)
n_cover = np.arange(*dist_cover_marg.ppf([0.001, 0.9999])).astype(int)
dat_pmf = pd.DataFrame({'x':n_cover, 'y':dist_cover_marg.pmf(n_cover)})
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
    # Add a plot of empirical coverage to beta-binomial
    gg_cover_class = (pn.ggplot(res_class, pn.aes(x='cover * n_val', y='..density..')) +
                      pn.theme_bw() + 
                      pn.geom_histogram(binwidth=1, color='blue', fill='grey', alpha=0.2) +
                      pn.labs(y='Density', x='Empirical coverage') + 
                      pn.geom_line(pn.aes(x='x',y='y'), data=dat_pmf, color='red') +
                      pn.ggtitle('Classification simulation\nRed line shows beta-binomial distribution'))
    gg_cover_class.save(os.path.join(dir_figs, 'cover_class.png'), width=6, height=4, verbose=False)
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
    res_reg = simulator.run_simulation(**sim_kwargs, verbose=True, n_iter=250)
    print('~~~ Regression intervals ~~~')
    print(f"CP: coverage={100*res_reg['cover'].mean():.1f}%, interval width={res_reg['set_size'].mean():.2f}, q={res_reg['qhat'].mean():.3f}")

    # Add a plot of empirical coverage to beta-binomial
    gg_cover_reg = (pn.ggplot(res_reg, pn.aes(x='cover * n_val', y='..density..')) +
                      pn.theme_bw() + 
                      pn.geom_histogram(binwidth=1, color='blue', fill='grey', alpha=0.2) +
                      pn.labs(y='Density', x='Empirical coverage') + 
                      pn.geom_line(pn.aes(x='x',y='y'), data=dat_pmf, color='red') +
                      pn.ggtitle('Regression simulation\nRed line shows beta-binomial distribution'))
    gg_cover_reg.save(os.path.join(dir_figs, 'cover_reg.png'), width=6, height=4, verbose=False)

    print('\n')


