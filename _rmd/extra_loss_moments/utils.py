# Modules
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

class dist_Ycond:
    def __init__(self, mu_Y, sigma_Y, sigma_X, rho, mu_X):
        """
        Function to calculate the a univariate gaussian of: Y | X, when:

        (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])

        
        Remember that Y | X = x ~ N(mu_Y + sigma_Y/sigma_X*rho*(x-mu_X), sigma2_Y*(1-rho^2))
        """
        self.mu_Y = mu_Y
        self.sigma_Y = sigma_Y
        self.sigma_X = sigma_X
        self.rho = rho
        self.mu_X = mu_X
        self.sigma2_Y = sigma_Y**2
    

    def __call__(self, x):
        loc = self.mu_Y + (self.sigma_Y / self.sigma_X) * self.rho * (np.array(x) - self.mu_X)
        scale = np.sqrt(self.sigma2_Y * (1 - self.rho ** 2))
        return stats.norm(loc=loc, scale=scale)

# Define all distributions that quality for valid BVNs
accepted_dists = (multivariate_normal_frozen, rv_continuous_frozen, rv_discrete_frozen, dist_Ycond)

def _is_int(x):
    """Checks that x is an integer"""
    return int(x) == x


def check_other(num_samples, seed, k_sd) -> None:
    if num_samples is not None:
        assert num_samples > 1, f'You must have num_samples > 1, not {num_samples}'
    if seed is not None:
        assert _is_int(seed), f'seed must be an integer, not {seed}'
        assert seed >= 0, f'seed must be positive, not {seed}'
    if k_sd is not None:
        assert k_sd > 0, f'k must be strictkly positive, not {k_sd}'

def input_checks(loss, 
                dist1, 
                num_samples = None, 
                seed = None, 
                dist2 = None, 
                k_sd = None
                ) -> None:
    """Makes sure that MCI/NumInt arguments are as expected, assertion checks only"""
    assert isinstance(loss, Callable), f'loss must be a callable function, not {type(loss)}'
    assert isinstance(dist1, accepted_dists), f'the first distribution must be of type rv_continuous_frozen, not {type(dist1)}'
    if num_samples is not None:
        assert num_samples > 1, f'You must have num_samples > 1, not {num_samples}'
    if seed is not None:
        assert _is_int(seed), f'seed must be an integer, not {seed}'
        assert seed >= 0, f'seed must be positive, not {seed}'
    if dist2 is not None:
        assert isinstance(dist2, Callable), f'the second distribution must a callable function, not {type(dist2)}'
    if k_sd is not None:
        assert k_sd > 0, f'k must be strictkly positive, not {k_sd}'


expected_dist = stats._multivariate.multivariate_normal_frozen
def generate_ellipse_points(
                            dist : expected_dist, 
                            alpha : float = 0.05, 
                            n_points : int=100,
                            ret_df : bool = False,
                            ) -> np.ndarray | pd.DataFrame:
    """
    Function to generate Gaussian condifence interval ellipse

    Parameters
    ----------
    dist : expected_dist
        The fitted multivariate normal distribution
    alpha : float, optional
        The quantile level (note that alpha < 0.5 means values closed to the centroid)
    n_points : int=100
        How many points to sample from the quantile-specific ellipse?
    ret_df : bool, optional
        Should a DataFrame be returned instead 
    """
    # Input checks
    assert isinstance(dist, expected_dist), f'dist needs to be {expected_dist}, not {type(dist)}'
    # Step 1: Exctract relevant terms
    mu = dist.mean
    cov = dist.cov 
    # Step 2: Calculate the radius of the ellipse
    chi2_val = stats.chi2.ppf(alpha, df=2)
    # Step 3: Parametric angle
    theta = np.linspace(0, 2 * np.pi, n_points)
    # Step 4: Parametric form of the ellipse
    ellipse = np.array([np.cos(theta), np.sin(theta)])
    # Step 5: Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)
    # Step 6: Scaling the ellipse by chi2_val
    ellipse_scaled = np.sqrt(chi2_val) * np.dot(L, ellipse)
    # Step 7: Translate the ellipse by the mean
    ellipse_points = ellipse_scaled.T + mu
    if ret_df:
        ellipse_points = pd.DataFrame(ellipse_points, columns = ['X1', 'X2'])
    return ellipse_points
