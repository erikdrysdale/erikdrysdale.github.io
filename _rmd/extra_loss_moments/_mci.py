"""
Utility scripts for monte carlo integration
"""

# External modules
import numpy as np
from typing import Callable
from scipy.stats import norm, multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
# Intenral modules
from .utils import input_checks as _input_checks



def mci_joint(loss : Callable, 
              dist_joint : multivariate_normal_frozen, 
              num_samples : int, 
              seed: int | None = None
              ) -> float:
    """
    Function to compute the integral using Monte Carlo integration:

    (y_i, x_i) ~ F_{Y, X}
    (1/n) \sum_{i=1}^n loss(y_i, x_i)

    Parameters
    ----------
    loss : Callable
        Some loss function loss(y=..., x=...)
    dist_joint : rv_continuous_frozen
        Some scipy.stats dist such that (Y, X) ~ dist_joint.rvs(...)
    num_samples : int
        How many samples to draw: dist_joint.rvs(num_samples, ...)?
    seed: int | None, optional
        Reproducability seed: dist_joint.rvs(..., random_state=seed)

    Returns
    -------
    float
        Average over the integrand of mean([loss(y_1, x_1), ..., loss(y_{num_samples}, x_{num_samples})])
    """
    # Input checks
    _input_checks(loss=loss, dist1=dist_joint, num_samples=num_samples, seed=seed)
    # Draw data
    y_samples, x_samples = dist_joint.rvs(num_samples, random_state=seed).T
    # Calculate the average of the integrand
    mu = np.mean(loss(y=y_samples, x=x_samples))
    return mu



def mci_cond(loss : Callable, 
              dist_X_uncond : rv_continuous_frozen, 
              dist_Y_condX : Callable, 
              num_samples : int, 
              seed: int | None = None
              ) -> float:
    """
    Function to compute the integral using Monte Carlo integration. NOTE! The way scipy implements the .rvs method, if the `random_state` and `n` are the same, it uses the same uniform number draw so we need to increment the seed

    X_i \sim F_X
    Y_i | X_i \sim F_{Y | X}
    (1/n) \sum_{i=1}^n loss(y_i, x_i)

    Parameters
    ----------
    dist_X_uncond : rv_continuous_frozen
        Some scipy.stats dist such that X ~ dist_X_uncond.rvs(...)
    dist_Y_condX : Callable
        Some function that returns a scipy.stats dist such that Y ~ dist_Y_condX(X=x).rvs(...)
    **kwargs
        For other named arugments, see mci_joint
    """
    # Input checks
    _input_checks(loss=loss, dist1=dist_X_uncond, 
                num_samples=num_samples, seed=seed, 
                dist2=dist_Y_condX)
    # Draw data in two steps
    x_samples = dist_X_uncond.rvs(num_samples, random_state=seed)
    y_samples = dist_Y_condX(x_samples).rvs(num_samples, random_state=seed+1)
    # Calculate the average of the integrand
    mu = np.mean(loss(y=y_samples, x=x_samples))
    return mu
