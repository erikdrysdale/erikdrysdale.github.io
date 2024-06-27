"""
Check for ratio adjustments
"""

import numpy as np
from scipy.stats import chi2, beta

# parameters
nsim = 500
sample_sizes = [5, 10, 25, 100, 250, 1000]

# Set up distributions
m = 10
z = 20
dist_Y = chi2(dof=m)
# dist_ratio = 

# population param
mu_ratio = 

for sample_size in sample_sizes:
    np.mean(dist_num.rvs(size=(sample_size, nsim)) / dist_den.rvs(size=(sample_size, nsim)), axis=0).mean()
    sample_size
