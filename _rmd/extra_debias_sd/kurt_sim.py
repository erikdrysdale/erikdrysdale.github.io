"""
CHECK ACCURACY OF KURSOSIS

python3 -m _rmd.extra_debias_sd.kurt_sim
"""

# Modules
import os
import numpy as np
import pandas as pd
from scipy.stats import moment, kurtosis, norm, expon
from _rmd.extra_debias_sd.utils import draw_from_data
from _rmd.extra_debias_sd.funs_kurt import kappa_from_moments

# Folders
dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '_rmd', 'extra_debias_sd')
dir_figs = os.path.join(dir_data, 'figures')

# Reproducabilty seed
seed = 1234

# empirical data
data = pd.read_csv(os.path.join(dir_data, 'data.csv'))['resid'].values


##########################
# --- (1) UNIT TESTS --- #

# Input data
np.random.seed(seed)
dims = (7, 3)
x = np.random.randn(*dims)

# (i) Compare standard kurtosis estimator
mdl_kurt = kappa_from_moments(x, debias=False, normal_adj=False)
np.testing.assert_almost_equal(mdl_kurt.kappa,
                               kurtosis(x, axis=0, fisher=False, bias=True))
mdl_kurt = kappa_from_moments(x, debias=False, normal_adj=True)
np.testing.assert_almost_equal(mdl_kurt.kappa,
                               kurtosis(x, axis=0, fisher=False, bias=False))

# (ii) Test for the LOO raw moments
# Calculate it manually
xbar_loo_manual = np.vstack([np.mean(np.delete(x, i, 0), axis=0) for i in range(x.shape[0])])
sigma2_loo_manual = np.vstack([np.var(np.delete(x, i, 0), axis=0) for i in range(x.shape[0])])
mu4_loo_manual = np.vstack([ moment(a=np.delete(x, i, 0), moment=4, axis=0) for i in range(x.shape[0])])
# Calculate with utility methods
xbar_loo = mdl_kurt._m1_loo(x, x.shape[0])
np.testing.assert_almost_equal(xbar_loo_manual, xbar_loo)
sigma2_loo = mdl_kurt._m2_loo(x, xbar_loo, np.mean(x**2, axis=0), x.shape[0])
np.testing.assert_almost_equal(sigma2_loo_manual, sigma2_loo)
mu4_loo = mdl_kurt._m4_loo(x, xbar_loo, np.mean(x**2, axis=0), 
                           np.mean(x**3, axis=0), np.mean(x**4, axis=0), x.shape[0])
np.testing.assert_almost_equal(mu4_loo_manual, mu4_loo)

# (iii) Calculate with model class
mdl_kurt = kappa_from_moments(x, debias=True, jacknife=True, store_loo=True)
np.testing.assert_almost_equal(xbar_loo_manual, mdl_kurt.m1_loo)
np.testing.assert_almost_equal(sigma2_loo_manual, mdl_kurt.m2_loo)
np.testing.assert_almost_equal(mu4_loo_manual, mdl_kurt.m4_loo)


########################################
# --- (2) CHECK UNBIASED STABILITY --- #

# Simulation parameters
nsim = 5000
sample_sizes = [10, 25, 100, 250, 1000]

# Clean up the "describe" columns
pcts = {0.1:'lb', 0.5:'med', 0.9:'ub'}
di_cols_describe = {'mean':'mu', 'std':'sd', 
                    'min':'mi', 'max':'mx'}
di_cols_describe = {**di_cols_describe,
                    **{f'{100*k:.0f}%':v for k,v in pcts.items()}}

# What distribution will be used?
# dist_sim = expon(scale=2)
dist_sim = draw_from_data(x = data)

# Run the for-loop simulation
holder = []
for sample_size in sample_sizes:
    print(f'Sample size: {sample_size}')
    # Draw data
    x = dist_sim.rvs(size=(sample_size, nsim), random_state=seed)
    # Calculate kappa and de-biased moments
    mdl_kappa = kappa_from_moments(x, debias=True)
    mdl_loo = kappa_from_moments(x, debias=True, jacknife=True)
    # Calculate raw moments
    xbar = np.mean(x, axis = 0)
    mu22_raw = np.var( x, axis=0) ** 2
    mu4_raw = np.mean( (x - xbar)**4, axis=0)
    kappa_raw = mu4_raw / mu22_raw
    # Combine
    tmp_df_adj = pd.DataFrame({'kappa':mdl_kappa.kappa, 
                               'mu4':mdl_kappa.mu4,
                               'mu22': mdl_kappa.mu22})
    tmp_df_raw = pd.DataFrame({'kappa':kappa_raw, 
                               'mu4':mu4_raw,
                               'mu22': mu22_raw})
    tmp_df = pd.concat(objs = [tmp_df_adj.assign(debias=True), 
                               tmp_df_raw.assign(debias=False)]).\
                    melt('debias', var_name='param').\
                        groupby(['debias','param'])['value'].\
                    describe(percentiles=list(pcts)).\
                    drop(columns='count').reset_index()
    tmp_df.insert(0, 'n', sample_size)
    holder.append(tmp_df)
# Merge and rename
res_components = pd.concat(holder).rename(columns=di_cols_describe)
# Add on "oracle" values
oracle_kurt = dist_sim.stats('k') + 3
oracle_mu22 = dist_sim.stats('v')**2
oracle_mu4 = oracle_kurt * oracle_mu22
dat_oracle = pd.DataFrame({'param':['kappa', 'mu22', 'mu4'],
              'oracle':[oracle_kurt, oracle_mu22, oracle_mu4]})
# Calculate as a "percent"
res_components_pct = res_components.set_index(['param', 'n', 'debias']).\
    divide(dat_oracle.set_index('param')['oracle'], axis=0).\
    reset_index()


########################
# --- (3) PLOTTING --- #

# Plotting functions
import plotnine as pn
from mizani.formatters import percent_format

gg_debiased_moments = (pn.ggplot(res_components_pct, pn.aes(x='n', y='mu')) + 
                 pn.theme_bw() + 
                 pn.geom_line(pn.aes(color='debias'), size=1) + 
                 pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub', fill='debias'), alpha=0.15, color='none') + 
                 pn.facet_wrap('~param', scales='free_y') + 
                 pn.scale_x_log10() + 
                 pn.scale_y_continuous(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.ggtitle('Ribbon shows simulation 80% CI') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Debias used?') + 
                 pn.scale_fill_discrete(name='Debias used?'))
gg_debiased_moments.save(os.path.join(dir_figs, 'debiased_moments.png'), height=3.5, width=10)

