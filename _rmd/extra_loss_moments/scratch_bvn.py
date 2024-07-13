"""
Make sure we can do integration for the bivariate normal dist. Specifically, if (Y,X) ~ BVN(mu, Sigma), can we estimate some function of: loss(Y,X)? In this example, loss(Y,X) = |Y| * log(X**2)

python3 -m _rmd.extra_loss_moments.scratch_poc
"""

# External modules
import numpy as np
import pandas as pd
from typing import Callable, Tuple
from scipy.integrate import dblquad
from scipy.stats import norm, multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

##############################
# --- (1) SET PARAMETERS --- #

# Define the loss function
def loss_function(y, x):
    return np.abs(y) * np.log(x**2)

# Bivariate normal parameters
mu_Y = 2.4
mu_X = -0.5
sigma2_Y = 2.1
sigma2_X = 0.9
rho = 0.7
mu = [mu_Y, mu_X]
sigma_Y = np.sqrt(sigma2_Y)
sigma_X = np.sqrt(sigma2_X)
rho = rho
cov_matrix = [[sigma_Y**2, rho * sigma_Y * sigma_X],
              [rho * sigma_Y * sigma_X, sigma_X**2]]
cov_matrix = np.array(cov_matrix)

# Create the distributions to draw on
dist_YX = multivariate_normal(mean=mu, cov=cov_matrix)
dist_Y = norm(loc=mu_Y, scale=np.sqrt(sigma2_Y))
dist_X = norm(loc=mu_X, scale=np.sqrt(sigma2_X))
dist_Yx = lambda x: norm(loc=mu_Y + (sigma_Y / sigma_X)*rho*(x - mu_X), scale=np.sqrt(sigma2_Y*(1-rho**2)))


################################################
# --- (2A) MONTE CARLO INTEGRATION (JOINT) --- #

# How many samples to draw frowm
num_samples = 2000000
seed = 1234

def _is_int(x):
    return int(x) == x

def _checks_mci(loss, dist1, 
                num_samples = None, seed = None, 
                dist2 = None, k_sd = None) -> None:
    """Makes sure that MCI arguments are as expected, assertion checks only"""
    accepted_dists = (multivariate_normal_frozen, rv_continuous_frozen, rv_discrete_frozen)
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
    _checks_mci(loss=loss, dist1=dist_joint, num_samples=num_samples, seed=seed)
    # Draw data
    y_samples, x_samples = dist_joint.rvs(num_samples, random_state=seed).T
    # Calculate the average of the integrand
    mu = np.mean(loss(y=y_samples, x=x_samples))
    return mu

# Calculate the integral using Monte Carlo integration
res_mci_joint = mci_joint(loss_function, dist_YX, num_samples, seed)
print(f"Monte Carlo Integration (joint): {res_mci_joint:.4f}")


######################################################
# --- (2B) MONTE CARLO INTEGRATION (CONDITIONAL) --- #

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
    _checks_mci(loss=loss, dist1=dist_X_uncond, 
                num_samples=num_samples, seed=seed, 
                dist2=dist_Y_condX)
    # Draw data in two steps
    x_samples = dist_X_uncond.rvs(num_samples, random_state=seed)
    y_samples = dist_Y_condX(x_samples).rvs(num_samples, random_state=seed+1)
    # Calculate the average of the integrand
    mu = np.mean(loss(y=y_samples, x=x_samples))
    return mu

# Calculate the integral using Monte Carlo integration
res_mci_cond = mci_cond(loss_function, dist_X, dist_Yx, num_samples, seed)
print(f"Monte Carlo Integration (conditional): {res_mci_cond:.4f}")


###################################################
# --- (2A) NUMERICAL INTEGRATION (JOINT) QUAD --- #

def _integrand_joint(y, x, loss, dist):
    """Internal function to compute the integrand"""
    loss_values = loss(y=y, x=x)
    density = dist.pdf([y, x])
    return loss_values * density

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

def numint_joint_quad(loss : Callable, 
                      dist_joint : multivariate_normal_frozen, 
                      k_sd : int = 4,
                      sol_tol : float = 1e-3
                      ):
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
    _checks_mci(loss=loss, dist1=dist_joint, k_sd=k_sd)
    y_min, y_max, x_min, x_max = _gen_bvn_bounds(dist_joint=dist_joint, k_sd=k_sd)
    # Calculate the integral using numerical integration (joint density)
    mu, _ = dblquad(_integrand_joint, x_min, x_max, y_min, y_max,
                    args=(loss, dist_joint), epsabs=sol_tol, epsrel=sol_tol)
    return mu

res_numint_joint_quad = numint_joint_quad(loss_function, dist_YX, sol_tol=1e-3, k_sd=4)
print(f"Numerical Integration (Joint) dblquad: {res_numint_joint_quad:.4f}")


####################################################
# --- (2B) NUMERICAL INTEGRATION (JOINT) TRAPZ --- #

def numint_joint_trapz(loss : Callable, 
                      dist_joint : multivariate_normal_frozen, 
                      k_sd : int = 3,
                      n_Y : int | np.ndarray = 100,
                      n_X : int | np.ndarray = 101,
                      use_grid : bool = False,
                      ) -> float:
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
    _checks_mci(loss=loss, dist1=dist_joint, k_sd=k_sd)
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
        loss_values = loss_function(Yvals, Xvals)
        density_values = dist_joint.pdf(np.dstack((Yvals, Xvals)))
        integrand_values = loss_values * density_values
        # Integrate out the Yvals
        inner_integral_X = np.trapz(integrand_values, yvals, axis=1)
    else:
        inner_integral_X = np.zeros(xvals.shape[0])
        for i, x_i in enumerate(xvals):
            inner_integrand = loss_function(yvals, x_i)
            inner_integrand *= dist_joint.pdf(np.c_[yvals, np.broadcast_to(x_i, yvals.shape)])
            inner_integral_X[i] = np.trapz(inner_integrand, yvals)
    # Calculate the outer integral by integrating the integrand (series of inner integrals) over X
    outer_integral = np.trapz(inner_integral_X, xvals)
    return outer_integral

res_numint_joint_trapz = numint_joint_trapz(loss=loss_function, dist_joint=dist_YX, use_grid=True)
assert numint_joint_trapz(loss=loss_function, dist_joint=dist_YX, use_grid=False) == res_numint_joint_trapz, 'use_grid should not change results'
print(f"Numerical Integration (Joint) trapezoidal: {res_numint_joint_trapz:.4f}")
# Compate run-time
from timeit import timeit
num_t = 250
t_grid = timeit("numint_joint_trapz(loss=loss_function, dist_joint=dist_YX, use_grid=False)", number=num_t, globals=globals())
t_loop = timeit("numint_joint_trapz(loss=loss_function, dist_joint=dist_YX, use_grid=True)", number=num_t, globals=globals())
print(f'Run time (s): grid={t_grid:.2f}, loop={t_loop:.2f}')


###################################################
# --- (2C) NUMERICAL INTEGRATION (COND) TRAPZ --- #


# (1) CREATE INTO CUSTOM FUNCTIONS (DOCSTRINGS FOR EVERYONE!)
# (2) RE-GENERALIZE TO DIST_X OUTSIDE OF THE BVN (I.E. IF F_X IS A MULTIVARIATE), SINCE BVN ONLY RELEVANT WHEN X ~ MVN, AND F_THETA(X): X'THETA


# Note that calling quad(outer_integrad, ...), outer_integrad = lambda x: quad(inner_intergrand,...) just takes WAY TOO LONG!! 

# Remember that X | Y = y ~ N(mu_X + sig_X/sig_Y*rho*(y-mu_Y), sig2_X*(1-rho^2))
#   The inner integral will be solving I_X(y) = \int_X ell(Y=y, x)*f_{X|Y}(x,Y=y) dx
#   In other words, the integral says, hey, once we know the value of Y, we plug this into out loss function and conditional density, and integrate over the domain of X
#   Next, we have numerical values of f(y) = I_X(y), so we simply integrate of the domain of Y:
#   \int_Y f(y) dy = \int_Y \int_X I(y, x) dy dx

# (a) Approach 1: Vectorize norm()
sig_X_cond = np.sqrt( sigma2_X*(1-rho**2) )
mu_X_cond = mu_X + (sigma_X / sigma_Y)*rho*(yvals - mu_Y)
dist_X_cond = norm(loc=mu_X_cond, scale=sig_X_cond)
inner_integral = np.trapz(loss_function(Yvals, Xvals) * dist_X_cond.pdf(np.atleast_2d(xvals).T), xvals, axis=0)
outer_integrand = inner_integral * dist_Y.pdf(yvals)
outer_integral = np.trapz(outer_integrand, yvals)
print(f"Trapezoidal Integration (Conditional Density) Result: {outer_integral:.4f}")


# (b) Approach 2: Loop over the norm moments
sig_X_cond = np.sqrt( sigma2_X*(1-rho**2) )
outer_integrand = np.zeros(num_y_points)
for j, y in enumerate(yvals):
    # Outer loop
    mu_X_cond = mu_X + (sigma_X / sigma_Y)*rho*(y - mu_Y)
    dist_X_cond = norm(loc=mu_X_cond, scale=sig_X_cond)
    # Solve inner integral
    inner_integrand = loss_function(y=y, x=xvals) * dist_X_cond.pdf(xvals)
    inner_integral = np.trapz(inner_integrand, xvals)
    outer_integrand[j] = inner_integral
outer_integrand *= dist_Y.pdf(yvals)
cond_density_trapz = np.trapz(outer_integrand, yvals)
print(f"Trapezoidal Integration (Conditional Density) Result: {cond_density_trapz:.4f}")


###############################
# --- (3) COMBINE RESULTS --- #

di_res = {
            'MCI Joint': res_mci_joint,
            'MCI Conditional': res_mci_cond,
            'NumInt Joint Quad': res_numint_joint_quad, 
            'NumInt Joint Trapz': res_numint_joint_trapz,
            'NumInt Conditional Trapz': None
        }
df_res = pd.DataFrame.from_dict(di_res, orient='index').\
    rename(columns={0:'integral'}).\
    rename_axis('method').\
    round(3).\
    reset_index()
print(df_res)

