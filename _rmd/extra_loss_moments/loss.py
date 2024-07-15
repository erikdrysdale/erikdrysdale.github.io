# External modules
import numpy as np
from typing import Callable, Tuple
from scipy.stats import norm, multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
# Intenral modules
from .utils import _is_int


class dist_Ycond:
    def __init__(self, mu_Y, sigma_Y, sigma_X, rho, mu_X):
        """
        Function to calculate the a univariate gaussian of: Y | X, when:

        (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])
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
        return norm(loc=loc, scale=scale)

class bvn_integral():
    def __init__(self) -> None:
        """
        Workhorse class for calculate the risk and loss variance of:

        R = E[loss(Y, X)]
        V = E[(loss(Y,X) - E[loss(Y, X)])^2]

        When (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])
        """
        pass

