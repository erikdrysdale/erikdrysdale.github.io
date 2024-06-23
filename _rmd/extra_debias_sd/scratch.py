"""
python3 -m _rmd.extra_debias_sd.scratch
"""
import plotnine as pn
import pandas as pd
import numpy as np


def sd_adj(x: np.ndarray, order:int, kappa: np.ndarray | None = None, ddof:int = 1, axis: int | None = None) -> np.ndarray:
    """
    
    """
    std = np.std(x, axis=axis, ddof = ddof)
    nrow = x.shape[0]
    if order == 1:
        assert kappa is not None, 'if order == 1, you have to supply kappa'
        adj = 1 / ( 1 - (kappa - 1 + 2/(nrow-1)) / (8*nrow) ) 
        std = std * adj
    return std
    

from scipy.stats import binom, kurtosis

seed = 2
m = 20
p_seq = [0.1, 0.3, 0.5]
n_p = len(p_seq)
dist_binom = binom(n=m, p=p_seq)
nsim = 100000
sample_sizes = np.arange(5, 50 + 1, 5)[:1]
# Calculate the population SD
dat_sd_binom = pd.DataFrame({'p':p_seq, 'sd':np.sqrt(dist_binom.stats('v'))})

# Loop over each sample size
holder_binom = []
for sample_size in sample_sizes:
    # Draw data
    X = dist_binom.rvs(size=[sample_size, nsim, n_p], random_state=seed)
    # Calculate different kappa variants
    p = X.mean(axis = 0) / m
    p[p == 0] = 1 / (m + 1)
    kappa_para =  3 - 6/m + 1/(m*p*(1-p)) 
    kappa_np_unj = kurtosis(a=X, fisher=False, bias=True, axis=0)
    kappa_np_adj = kurtosis(a=X, fisher=False, bias=False, axis=0)
    # Estimate the SD
    sigma_vanilla = sd_adj(x=X, order=0, axis=0)
    sigma_para = sd_adj(x=X, order=1, axis=0, kappa=kappa_para)
    sigma_unj = sd_adj(x=X, order=1, axis=0, kappa=kappa_np_unj)
    sigma_adj = sd_adj(x=X, order=1, axis=0, kappa=kappa_np_adj)


print('~~~ End of scratch ~~~')