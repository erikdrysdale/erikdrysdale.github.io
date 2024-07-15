"""
Utility scripts for numerical methods
"""

# External modules
import numpy as np
from typing import Callable, Tuple
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen
# Intenral modules
from .utils import _is_int
from .utils import input_checks as _input_checks


def _gen_bvn_bounds(dist_joint : multivariate_normal_frozen,
                    k_sd : int
                    ) -> Tuple[float, float, float, float]:
    """
    For each unconditional distribution, calculates the +/- k_sd*sd away from the mean
    
    """
    yx_bounds = np.atleast_2d(dist_joint.mean) + np.tile([-1,1],[2,1]).T * k_sd * np.sqrt(np.diag(dist_joint.cov))
    y_min, y_max = yx_bounds.T[0]
    x_min, x_max = yx_bounds.T[1]
    return y_min, y_max, x_min, x_max


def _integrand_joint(y, x, loss, dist):
    """Internal function to compute the integrand (for numint_joint_quad)"""
    loss_values = loss(y=y, x=x)
    density = dist.pdf([y, x])
    return loss_values * density

def _integrand2_joint(y, x, loss, dist):
    """Internal function to compute the integrand (for numint_joint_quad)"""
    loss_values = loss(y=y, x=x) ** 2
    density = dist.pdf([y, x])
    return loss_values * density


def numint_joint_quad(loss : Callable, 
                      dist_joint : multivariate_normal_frozen, 
                      k_sd : int = 4,
                      sol_tol : float = 1e-3,
                      calc_variance : bool = False,
                      ) -> float | Tuple[float, float]:
    """
    Calculates the integral of a l(y,x) assuming (y,x) ~ BVN using scipy's wrapper around QUADPACK

    Parameters
    ----------
    k_sd : int, optional
        How many standard deviations away from the mean to perform the grid search over?
    sol_tol : float, optional
        The values of `epsabs` & `epsrel` to pass into `scipy.integrate.dblquad`
    **kwargs
        See mci_joint
    """
    # Input checks and integration bounds
    _input_checks(loss=loss, dist1=dist_joint, k_sd=k_sd)
    y_min, y_max, x_min, x_max = _gen_bvn_bounds(dist_joint=dist_joint, k_sd=k_sd)
    # Calculate the integral using numerical integration (joint density)
    mu, _ = dblquad(_integrand_joint, x_min, x_max, y_min, y_max,
                    args=(loss, dist_joint), epsabs=sol_tol, epsrel=sol_tol)
    if calc_variance:
        loss2, _ = dblquad(_integrand2_joint, x_min, x_max, y_min, y_max,
                    args=(loss, dist_joint), epsabs=sol_tol, epsrel=sol_tol)
        var = loss2 - mu**2
        return mu, var
    else:
        return mu



def numint_joint_trapz(loss : Callable, 
                      dist_joint : multivariate_normal_frozen, 
                      k_sd : int = 4,
                      n_Y : int | np.ndarray = 100,
                      n_X : int | np.ndarray = 101,
                      use_grid : bool = False,
                      calc_variance : bool = False,
                      ) -> float | Tuple[float, float]:
    """
    Calculates the integral of a l(y,x) assuming (y,x) ~ BVN, using a double integral approach and the trapezoidal rule. 
    
    Description
    -----------
    We are solving the I_{YX} = \int_Y \int_X l(y,x) f_{YX}(y,x) dx dy. We do this in two steps: (1) First, for each x_i in {x_1, ..., x_{n_X}}, calculate the inner integral: I_{inner}(x_i) = \int_Y l(y,X=x_i) f_{YX}(y,X=x_i) dy. This gets back an (n_X,) length arrary. Next, calculate the outer integral: I_{outer} = \int_X I_{inner}(x) dx.
    
    Parameters
    ----------
    k_sd : int, optional
        How many standard deviations away from the mean to perform the grid search over?
    n_{Y,X} : int | np.ndarray, optional
        The number of points along the Y, X direction to generate (equal spacing). If an array is provided, will assume these are the points to use for integration
    use_grid : bool, optional
        Whether a grid a values should be used, or whether we loop over X
    **kwargs
        See mci_joint
    """
    # Input checks
    _input_checks(loss=loss, dist1=dist_joint, k_sd=k_sd)
    has_Y = not _is_int(n_Y)
    has_X = not _is_int(n_X)
    assert (has_Y & has_X) or (not has_Y and not has_X), f'if n_X is provided as an array, n_Y must be as well'
    # Set up integration bounds
    if not has_Y:
        y_min, y_max, x_min, x_max = _gen_bvn_bounds(dist_joint=dist_joint, k_sd=k_sd)
        yvals = np.linspace(y_min, y_max, n_Y)
        xvals = np.linspace(x_min, x_max, n_X)
    else:
        assert isinstance(n_Y, np.ndarray), f'if n_Y is not an int, it should be an array'
        assert isinstance(n_X, np.ndarray), f'if n_X is not an int, it should be an array'
        yvals, xvals = n_Y, n_X
    # Calculate integrand and integral
    if use_grid:
        Yvals, Xvals = np.meshgrid(yvals, xvals)
        # Calculate the integrand
        loss_values = loss(Yvals, Xvals)
        density_values = dist_joint.pdf(np.dstack((Yvals, Xvals)))
        integrand_values = loss_values * density_values
        # Integrate out the Yvals
        inner_integral_X_mu = np.trapz(integrand_values, yvals, axis=1)
        if calc_variance:
            inner_integral_X_var = np.trapz(loss_values**2 * density_values, yvals, axis=1)
    else:
        inner_integral_X_mu = np.zeros(xvals.shape[0])
        for i, x_i in enumerate(xvals):
            inner_integrand_mu = loss(yvals, x_i)
            points_i = np.c_[yvals, np.broadcast_to(x_i, yvals.shape)]
            inner_integrand_mu *= dist_joint.pdf(points_i)
            inner_integral_X_mu[i] = np.trapz(inner_integrand_mu, yvals)
        if calc_variance:
            inner_integral_X_var = np.zeros(xvals.shape[0])
            for i, x_i in enumerate(xvals):
                inner_integrand_var = loss(yvals, x_i)**2
                points_i = np.c_[yvals, np.broadcast_to(x_i, yvals.shape)]
                inner_integrand_var *= dist_joint.pdf(points_i)
                inner_integral_X_var[i] = np.trapz(inner_integrand_var, yvals)
    # Calculate the outer integral by integrating the integrand (series of inner integrals) over X
    outer_integral_mu = np.trapz(inner_integral_X_mu, xvals)
    if calc_variance:
        outer_integral_var = np.trapz(inner_integral_X_var, xvals)
        outer_integral_var -= outer_integral_mu**2
        return outer_integral_mu, outer_integral_var
    else:
        return outer_integral_mu


def _gen_bvn_cond_bounds(
                        dist_Y_condX : Callable,
                        k_sd : int
                        ) -> Tuple[float, float, float, float]:
    """Re-construct the joint distribution and feed it into _gen_bvn_bounds(...)"""
    mu = np.array([dist_Y_condX.mu_Y, dist_Y_condX.mu_X])
    off_diag = dist_Y_condX.rho * dist_Y_condX.sigma_X * dist_Y_condX.sigma_Y
    cov = np.array([[dist_Y_condX.sigma_Y**2, off_diag], [off_diag, dist_Y_condX.sigma_X**2]])
    dist_joint = multivariate_normal(mean=mu, cov=cov)
    return _gen_bvn_bounds(dist_joint, k_sd)


def numint_cond_trapz(loss : Callable, 
                      dist_X_uncond : rv_continuous_frozen, 
                      dist_Y_condX : Callable, 
                      k_sd : int = 4,
                      n_Y : int | np.ndarray = 100,
                      n_X : int | np.ndarray = 101,
                      use_grid : bool = False,
                      calc_variance : bool = False,
                      ) -> float | Tuple[float, float]:
    """
    Calculates the integral of a l(y,x) assuming (y,x) ~ BVN, using a double integral approach and the trapezoidal rule. 
    
    Description
    -----------
    We are solving the I_{YX} = \int_Y \int_X l(y,x) f_{YX}(y,x) dx dy = \int_X [\int_Y l(y, X=x) f_{Y|X}(y,X=x) dy] f_X(x) dx.

    We do this in two steps: (1) First, for each x_i in {x_1, ..., x_{n_X}}, calculate the inner integral: I_{inner}(x_i) = \int_Y l(y,X=x_i) f_{Y|X}(y,X=x_i) dy. This gets back an (n_X,) length arrary. Next, calculate the outer integral: I_{outer} = \int_X I_{inner}(x) f_X(x) dx. 

    In essence, we're iterating over the "feature space" one point at a time, calculating the expected loss at that feature space point, and then doing a weighted sum by the likelihood of those feature points
    
    Parameters
    ----------
    k_sd : int, optional
        How many standard deviations away from the mean to perform the grid search over?
    n_{Y,X} : int | np.ndarray, optional
        The number of points along the Y, X direction to generate (equal spacing). If an array is provided, will assume these are the points to use for integration
    use_grid : bool, optional
        Whether a grid a values should be used, or whether we loop over X
    **kwargs
        See mci_cond
    """
    # Input checks
    _input_checks(loss=loss, dist1=dist_X_uncond, k_sd=k_sd, dist2=dist_Y_condX)
    has_Y = not _is_int(n_Y)
    has_X = not _is_int(n_X)
    assert (has_Y & has_X) or (not has_Y and not has_X), f'if n_X is provided as an array, n_Y must be as well'
    # Set up integration bounds
    if not has_Y:
        y_min, y_max, x_min, x_max = _gen_bvn_cond_bounds(dist_Y_condX=dist_Y_condX, k_sd=k_sd)
        yvals = np.linspace(y_min, y_max, n_Y)
        xvals = np.linspace(x_min, x_max, n_X)
    else:
        assert isinstance(n_Y, np.ndarray), f'if n_Y is not an int, it should be an array'
        assert isinstance(n_X, np.ndarray), f'if n_X is not an int, it should be an array'
        yvals, xvals = n_Y, n_X
    # Calculate integrand and integral
    if use_grid:
        Yvals, Xvals = np.meshgrid(yvals, xvals)
        # Calculate the integrand
        loss_values = loss(Yvals, Xvals)
        density_cond = dist_Y_condX(Xvals).pdf(Yvals)
        # Integrate out the Yvals
        inner_integral_X_mu = np.trapz(loss_values * density_cond, yvals, axis=1)
        if calc_variance:
            inner_integral_X_var = np.trapz(loss_values**2 * density_cond, yvals, axis=1)
    else:
        inner_integral_X_mu = np.zeros(xvals.shape[0])
        for i, x_i in enumerate(xvals):
            inner_integrand_mu = loss(yvals, x_i)
            inner_integrand_mu *= dist_Y_condX(x_i).pdf(yvals)
            inner_integral_X_mu[i] = np.trapz(inner_integrand_mu, yvals)
        if calc_variance:
            inner_integral_X_var = np.zeros(xvals.shape[0])
            for i, x_i in enumerate(xvals):
                inner_integrand_var = loss(yvals, x_i)**2
                inner_integrand_var *= dist_Y_condX(x_i).pdf(yvals)
                inner_integral_X_var[i] = np.trapz(inner_integrand_var, yvals)
    # Calculate the outer integral by integrating the integrand (series of inner integrals) over X
    outer_integrand_mu = inner_integral_X_mu * dist_X_uncond.pdf(xvals)
    outer_integral_mu = np.trapz(outer_integrand_mu, xvals)
    if calc_variance:
        outer_integrand_var = inner_integral_X_var * dist_X_uncond.pdf(xvals)
        outer_integral_var = np.trapz(outer_integrand_var, xvals)
        outer_integral_var -= outer_integral_mu**2
        return outer_integral_mu, outer_integral_var
    else:
        return outer_integral_mu