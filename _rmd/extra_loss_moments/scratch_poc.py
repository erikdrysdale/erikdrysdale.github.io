"""
Make sure we can do integration for the bivariate normal dist. Specifically, if (Y,X) ~ BVN(mu, Sigma), can we estimate some function of: loss(Y,X)? In this example, loss(Y,X) = |Y| * log(X**2)

python3 -m _rmd.extra_loss_moments.scratch_poc
"""

# External modules
import numpy as np
from typing import Callable
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
mu_Y = 1.4
mu_X = -0.5
sigma2_Y = 2.1
sigma2_X = 0.9
rho = 0.5
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

def _checks_mci(loss, dist1, num_samples, seed, dist2 = None):
    """
    Makes sure that MCI arguments are as expected
    """
    accepted_dists = (multivariate_normal_frozen, rv_continuous_frozen, rv_discrete_frozen)
    assert isinstance(loss, Callable), f'loss must be a callable function, not {type(loss)}'
    assert isinstance(dist1, accepted_dists), f'the first distribution must be of type rv_continuous_frozen, not {type(dist1)}'
    assert num_samples > 1, f'You must have num_samples > 1, not {num_samples}'
    if seed is not None:
        assert int(seed) == seed, f'seed must be an integer, not {seed}'
        assert seed >= 0, f'seed must be positive, not {seed}'
    if dist2 is not None:
        assert isinstance(dist2, Callable), f'the second distribution must a callable function, not {type(dist2)}'



def mci_joint(loss : Callable, 
              dist_joint : rv_continuous_frozen, 
              num_samples : int, 
              seed: int | None = None):
    """
    Function to compute the integral using Monte Carlo integration:

    (y_i, x_i) ~ F_{Y, X}
    (1/n) \sum_{i=1}^n loss(y_i, x_i)

    Parameters
    ----------
    """
    # Input checks
    _checks_mci(loss, dist_joint, num_samples, seed)
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

def mci_cond(loss, dist_X_uncond, dist_Y_condX, num_samples, seed = None):
    """
    Function to compute the integral using Monte Carlo integration. NOTE! The way scipy implements the .rvs method, if the `random_state` and `n` are the same, it uses the same uniform number draw so we need to increment the seed

    X_i \sim F_X
    Y_i | X_i \sim F_{Y | X}
    (1/n) \sum_{i=1}^n loss(y_i, x_i)
    """
    # Input checks
    _checks_mci(loss, dist_X_uncond, num_samples, seed, dist_Y_condX)
    # Draw data in two steps
    x_samples = dist_X_uncond.rvs(num_samples, random_state=seed)
    y_samples = dist_Y_condX(x_samples).rvs(num_samples, random_state=seed+1)
    # Calculate the average of the integrand
    mu = np.mean(loss(y=y_samples, x=x_samples))
    return mu

# Calculate the integral using Monte Carlo integration
res_mci_cond = mci_cond(loss_function, dist_X, dist_Yx, num_samples, seed)
print(f"Monte Carlo Integration (conditional): {res_mci_cond:.4f}")


#############################################
# --- (2) NUMERICAL INTEGRATION (JOINT) --- #

# (1) CREATE INTO CUSTOM FUNCTIONS

def numint_joint_quad(loss, dist_joint, num_samples):
    return None

# Function to compute the integrand
def integrand_joint(y, x, loss, dist):
    loss_values = loss(y=y, x=x)
    density = dist.pdf([y, x])
    return loss_values * density

# Bounds for integration
alpha = 1e-4
alphas = np.array([alpha/2,1-alpha/2])
y_min, y_max = dist_Y.ppf(alphas)
x_min, x_max = dist_X.ppf(alphas)

# Calculate the integral using numerical integration (joint density)
sol_tol = 1e-3
res_numint_joint_quad, _ = dblquad(integrand_joint, x_min, x_max, y_min, y_max,
                                  args=(loss_function, dist_YX), epsabs=sol_tol, epsrel=sol_tol)
print(f"Numerical Integration (Joint) dblquad: {res_numint_joint_quad:.4f}")

# Repeat with trapezoidal rule
num_y_points = 50
num_x_points = 188
yvals = np.linspace(y_min, y_max, num_y_points)
xvals = np.linspace(x_min, x_max, num_x_points)
Yvals, Xvals = np.meshgrid(yvals, xvals)
# Calculate the integrand
loss_values = loss_function(Yvals, Xvals)
density_values = dist_YX.pdf(np.dstack((Yvals, Xvals)))
integrand_values = loss_values * density_values

# Perform numerical integration using the trapezoidal rule
#   Integrates x out conditional on each value of Y.
#   Remember, Yvals is constant over the axis, so axis=0 perfoms trapz(integrand[:,j], xvals)
#   i.e. the inner integral result_y[j] = [\int_{X} I(Y[j], x) dx] 
result_y = np.trapz(integrand_values, xvals, axis=0)
# Next, f(y) = \int_{X} \ell(Y=y, x) dx, and we do a single-variable integration of y:
#   \int_Y f(y) dy = \int_Y \int_X I(y, x) dy dx
res_numint_joint_trapz = np.trapz(result_y, yvals)
print(f"Numerical Integration (Joint) trapezoidal: {res_numint_joint_trapz:.4f}")


"""
(3) CONDITIONAL INTEGRAL
"""

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

