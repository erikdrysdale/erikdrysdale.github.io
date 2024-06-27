"""
CHECK ACCURACY OF KURSOSIS
"""


import numpy as np
from scipy.stats import moment

# Input data
np.random.seed(1234)
dims = (7, 3)
x = np.random.randn(*dims)

# difference in means is easy...
# \bar{x}_{-i} = \frac{n}{n-1} ( \bar{x} - x/n )
n = x.shape[0]
xbar = np.mean(x, axis=0)
sigma2 = np.var(x, axis=0)
sum_x = np.sum(x, axis=0)
sum_x3 = np.sum(x ** 3, axis=0)
sum_x2 = np.sum(x ** 2, axis=0)
sum_x4 = np.sum(x ** 4, axis=0)
mu_x = np.mean(x, axis=0)
mu_x3 = np.mean(x ** 3, axis=0)
mu_x2 = np.mean(x ** 2, axis=0)
mu_x4 = np.mean(x ** 4, axis=0)


# CHECK ASSERTIONS
xbar_loo = (n*xbar - x)/(n-1)
xbar_loo_manual = np.vstack([np.mean(np.delete(x, i, 0), axis=0) for i in range(n)])
np.testing.assert_almost_equal(xbar_loo, xbar_loo_manual)
sigma2_loo = (sum_x2 - x ** 2 - (n - 1) * xbar_loo ** 2) / (n - 1)
sigma2_loo_manual = np.vstack([np.var(np.delete(x, i, 0), axis=0) for i in range(n)])
np.testing.assert_almost_equal(sigma2_loo, sigma2_loo_manual)
# Based on the expansion of the fourth central moment...
mu4_loo = ( 
            (sum_x4 - x ** 4) 
            - 4 * xbar_loo * (sum_x3 - x ** 3) 
            + 6 * xbar_loo ** 2 * (sum_x2 - x ** 2) 
            - 3 * (n-1)* xbar_loo ** 4
            )  / (n - 1)
mu4_loo_manual = np.vstack([ moment(a=np.delete(x, i, 0), moment=4, axis=0) for i in range(n)])
np.testing.assert_almost_equal(mu4_loo, mu4_loo_manual)

# CAN WE CONVERT THIS TO MEAN_X{J} FOR NUMERICAL STABILITIY???
sigma2_loo_alt = n*(mu_x2 - x**2 / n - (n - 1) * xbar_loo**2 / n) / (n - 1)
np.testing.assert_almost_equal(sigma2_loo_alt, sigma2_loo_manual)
# Based on the expansion of the fourth central moment...
mu4_loo_alt = n*( 
            (mu_x4 - x**4/n) 
            - 4 * xbar_loo * (mu_x3 - x**3/n) 
            + 6 * xbar_loo ** 2 * (mu_x2 - x**2/n) 
            - 3 * (n-1)* xbar_loo ** 4 / n
            )  / (n - 1)
np.testing.assert_almost_equal(mu4_loo_alt, mu4_loo_manual)




# Define functions equivalent to Umoments::uM4 and Umoments::uM2pow2
def uM4(m2, m4, n):
    term1 = -3 * m2**2 * (2 * n - 3) * n / ((n - 1) * (n - 2) * (n - 3))
    term2 = (n**2 - 2 * n + 3) * m4 * n / ((n - 1) * (n - 2) * (n - 3))
    return term1 + term2

def uM2pow2(m2, m4, n):
    term1 = (n**2 - 3 * n + 3) * m2**2 * n / ((n - 1) * (n - 2) * (n - 3))
    term2 = -m4 * n / ((n - 2) * (n - 3))
    return term1 + term2


# Wrapper function
def kappa_from_moments(
        x: np.ndarray, 
        ret_all:bool = False, 
        loo: bool = False
    ):
    # Process data
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[0]
    
    # Compute raw central moments
    m1 = np.mean(x, axis=0)
    m2 = np.mean( (x - m1)**2, axis=0)
    m4 = np.mean( (x - m1)**4, axis=0)
    if loo:
        # Apply the LOO correction
        None
    
    # Compute adjusted moments
    uM4_result = uM4(m2, m4, n)
    uM2pow2_result = uM2pow2(m2, m4, n)
    kappa = uM4_result / uM2pow2_result

    # Return 
    if ret_all:
        return kappa, uM4_result, uM2pow2_result
    else:
        return kappa


kappa_from_moments(x).mean()
