"""
Make sure we can do integration for the bivariate normal dist

python3 -m _rmd.extra_loss_moments.scratch1
"""


# External modules
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.integrate import quad
from scipy.integrate import dblquad
# Internal modules
from _rmd.extra_loss_moments.utils import generate_ellipse_points


# Parameters for the bivariate normal distribution
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
dist_mvn = stats.multivariate_normal(mean=mu, cov=cov_matrix)
dist_Y = norm(loc=mu_Y, scale=np.sqrt(sigma2_Y))
dist_X = norm(loc=mu_X, scale=np.sqrt(sigma2_X))


def loss_function(y, x):
    return np.abs(y) * np.log(x**2)


"""
(1) MONTE CARLO
"""

# Function to compute the integral using Monte Carlo integration
def monte_carlo_integration(loss, dist, num_samples, seed = None):
    samples = dist.rvs(num_samples, random_state=seed)
    y_samples = samples[:, 0]
    x_samples = samples[:, 1]
    mu_integrand_values = np.mean(loss(y=y_samples, x=x_samples))
    return mu_integrand_values

# Calculate the integral using Monte Carlo integration
num_samples = 1000000
monte_carlo_result = monte_carlo_integration(loss_function, dist_mvn, num_samples)
print(f"Monte Carlo Integration Result: {monte_carlo_result:.4f}")


"""
(2) JOINT INTEGRAL
"""

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
joint_density_result, _ = dblquad(integrand_joint, x_min, x_max, y_min, y_max,
                                  args=(loss_function, dist_mvn), epsabs=sol_tol, epsrel=sol_tol)
print(f"Numerical Integration (Joint Density) Result: {joint_density_result:.4f}")

# Repeat with trapezoidal rule
num_y_points = 50
num_x_points = 188
yvals = np.linspace(y_min, y_max, num_y_points)
xvals = np.linspace(x_min, x_max, num_x_points)
Yvals, Xvals = np.meshgrid(yvals, xvals)
# Calculate the integrand
loss_values = loss_function(Yvals, Xvals)
density_values = dist_mvn.pdf(np.dstack((Yvals, Xvals)))
integrand_values = loss_values * density_values

# Perform numerical integration using the trapezoidal rule
#   Integrates x out conditional on each value of Y.
#   Remember, Yvals is constant over the axis, so axis=0 perfoms trapz(integrand[:,j], xvals)
#   i.e. the inner integral result_y[j] = [\int_{X} I(Y[j], x) dx] 
result_y = np.trapz(integrand_values, xvals, axis=0)
# Next, f(y) = \int_{X} \ell(Y=y, x) dx, and we do a single-variable integration of y:
#   \int_Y f(y) dy = \int_Y \int_X I(y, x) dy dx
joint_density_trapz = np.trapz(result_y, yvals)
print(f"Trapezoidal Integration (Joint Density) Result: {joint_density_trapz:.4f}")


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

