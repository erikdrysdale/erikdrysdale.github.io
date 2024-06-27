"""
CHECK ACCURACY OF KURSOSIS

python3 -m _rmd.extra_debias_sd.kurt_sim
"""


import numpy as np
from scipy.stats import moment, kurtosis
from _rmd.extra_debias_sd.funs_kurt import kappa_from_moments

# Input data
np.random.seed(1234)
dims = (7, 3)
x = np.random.randn(*dims)


##########################
# --- (1) UNIT TESTS --- #

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




import sys
sys.exit('stop here')
