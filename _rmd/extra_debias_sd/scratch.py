"""
python3 -m _rmd.extra_debias_sd.scratch
"""

import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import kurtosis
from _rmd.extra_debias_sd.utils import sd_adj, efficient_loo_kurtosis

#########################
# --- (1) LOAD DATA --- #

# load data
dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '_rmd', 'extra_debias_sd')
x = pd.read_csv(os.path.join(dir_data, 'data.csv'))['resid'].values

# full-sample kurtosis
n = len(x)
kappa_full = kurtosis(x, fisher=False, bias=False)

# generate log-scale sample samples
sample_sizes = np.exp(np.linspace(np.log(15), np.log(n), 20)).round().astype(int)


#######################################
# --- (2) SIMULATE FOR SUBSAMPLES --- #

# Comparse the different terms
nsim = 111
n_bs = 250
holder_sd = []
holder_kappa = []
holder_components = []
np.random.seed(nsim)
for sample_size in sample_sizes:
    # --- (i) Generate subsamples --- #
    if sample_size == n:
        bs_len = 1
        idx = np.arange(n)
    else:
        bs_len = nsim
        idx = np.vstack([np.random.choice(a=n, size=sample_size, replace=False) for _ in range(nsim)]).T
    x_idx = pd.DataFrame(x[idx])
    
    # --- (ii) Generate kurtosis components --- #
    xbar_idx = x_idx.mean(axis = 0)
    mu4_idx = (x_idx - xbar_idx).pow(4).mean(axis=0)
    sigma2_idx = x_idx.var(axis=0, ddof=1)
    sigma4_idx = sigma2_idx**2

    di_components = {'sigma4': sigma4_idx.mean(), 
                     'mu4': mu4_idx.mean()}
    tmp_df_components = pd.DataFrame.from_dict(di_components, orient='index').reset_index()
    tmp_df_components.insert(0, 'n', sample_size)
    holder_components.append(tmp_df_components)    

    break

    # --- (iii) Generate kurtosis estimates --- #
    kappa_adj = x_idx.kurtosis(axis = 0) + 3
    
    # Bias-adjusted bootstrap
    kappa_bs = np.zeros(bs_len)
    for j in range(n_bs):
        kurt_j = x_idx.\
            sample(frac=1, replace=True, axis=0, random_state=j+1).\
            kurtosis(axis = 0) + 3
        kappa_bs += kurt_j
    kappa_bs /= n_bs
    kappa_bs = 2*kappa_adj - kappa_bs
    # Bias-adjusted LOO
    kappa_loo = sample_size*kappa_adj - (sample_size - 1)*efficient_loo_kurtosis(x_idx).mean()
    di_kappa = {'adj':kappa_adj.mean(),
                'bs':kappa_bs.mean(),
                'loo':kappa_loo.mean()}
    tmp_df_kappa = pd.DataFrame.from_dict(di_kappa, orient='index').reset_index()
    tmp_df_kappa.insert(0, 'n', sample_size)
    holder_kappa.append(tmp_df_kappa)    
    
    # --- (iv) Generated SD estimates --- #
    mu_std_vanilla = x_idx.std(axis=0, ddof=1)
    mu_std_kappa_adj = sd_adj(x_idx, axis=0, ddof=1, kappa=kappa_adj)
    mu_std_kappa_full = sd_adj(x_idx, axis=0, ddof=1, kappa=kappa_full)
    di_std = {'vanilla':mu_std_vanilla.mean(),
            'kappa_adj':mu_std_kappa_adj.mean(),
            'kappa_full':mu_std_kappa_full.mean()}
    tmp_df_std = pd.DataFrame.from_dict(di_std, orient='index').reset_index()
    tmp_df_std.insert(0, 'n', sample_size)
    holder_sd.append(tmp_df_std)    
# Merge
res_sd = pd.concat(holder_sd).rename(columns={'index':'method', 0:'sd'}).reset_index(drop=True)
res_kappa = pd.concat(holder_kappa).rename(columns={'index':'method', 0:'kappa'}).reset_index(drop=True)
res_components = pd.concat(holder_components).rename(columns={'index':'metric', 0:'val'}).reset_index(drop=True)


###########################
# --- (3) POT RESULTS --- #

# Plot kappa components
components_oracle = res_components.loc[res_components.groupby('metric')['n'].idxmax()].groupby('metric')['val'].mean().reset_index().rename(columns={'val':'oracle'})
gg_components = (pn.ggplot(res_components, pn.aes(x='n', y='val', color='metric')) + 
                   pn.theme_bw() + pn.geom_line() + 
                   pn.facet_wrap('~metric',scales='free_y') + 
                   pn.guides(color=False) + 
                   pn.geom_hline(pn.aes(yintercept='oracle'), linetype='--',data=components_oracle) + 
                   pn.scale_x_log10(limits=[10, n]))
gg_components.save(os.path.join(dir_data, 'components_comp.png'), width=8, height=3.5)



# Plot kappa results
kappa_oracle = res_kappa.query('n == n.max()')['kappa'].mean()
gg_kappa = (pn.ggplot(res_kappa, pn.aes(x='n', y='kappa', color='method')) + 
                   pn.theme_bw() + pn.geom_line() + 
                   pn.geom_hline(yintercept=kappa_oracle, linetype='--') + 
                   pn.scale_x_log10(limits=[10, n]))
gg_kappa.save(os.path.join(dir_data, 'kappa_comp.png'), width=5, height=3.5)

# Plot SD results
sd_oracle = res_sd.query('n == n.max()')['sd'].mean()
gg_sd = (pn.ggplot(res_sd, pn.aes(x='n', y='sd', color='method')) + 
                   pn.theme_bw() + pn.geom_line() + 
                   pn.geom_hline(yintercept=sd_oracle, linetype='--') + 
                   pn.scale_x_log10(limits=[10, n]))
gg_sd.save(os.path.join(dir_data, 'sd_comp.png'), width=5, height=3.5)



print('~~~ End of scratch ~~~')