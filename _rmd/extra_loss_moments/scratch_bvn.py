"""
Make sure we can do integration for the bivariate normal dist. Specifically, if (Y,X) ~ BVN(mu, Sigma), can we estimate some function of: loss(Y,X)? In this example, loss(Y,X) = |Y| * log(X**2)

python3 -m _rmd.extra_loss_moments.scratch_bvn
"""

# External modules
import numpy as np
import pandas as pd
from timeit import timeit
from scipy.stats import norm, multivariate_normal
# Internal modules
from _rmd.extra_loss_moments.loss import dist_Ycond
from _rmd.extra_loss_moments._mci import mci_cond, mci_joint
from _rmd.extra_loss_moments._numerical import numint_joint_quad, numint_joint_trapz, numint_cond_trapz

# Run-time iterations
num_t = 200

##############################
# --- (1) SET PARAMETERS --- #

# Define the loss function
def loss_function(y, x):
    return np.abs(y) * np.log(x**2)

# Bivariate normal parameters
mu_Y = 2.4
mu_X = -3.5
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
dist_Yx = dist_Ycond(mu_Y=mu_Y, sigma_Y=sigma_Y, sigma_X=sigma_X, rho=rho, mu_X=mu_X)


################################################
# --- (2A) MONTE CARLO INTEGRATION (JOINT) --- #

# How many samples to draw frowm
num_samples = 2000000
seed = 12345

# Calculate the integral using Monte Carlo integration
res_mci_joint = mci_joint(loss_function, dist_YX, num_samples, seed)
print(f"Monte Carlo Integration (joint): {res_mci_joint:.4f}")


######################################################
# --- (2B) MONTE CARLO INTEGRATION (CONDITIONAL) --- #


# Calculate the integral using Monte Carlo integration
res_mci_cond = mci_cond(loss_function, dist_X, dist_Yx, num_samples, seed)
print(f"Monte Carlo Integration (conditional): {res_mci_cond:.4f}")


###################################################
# --- (2A) NUMERICAL INTEGRATION (JOINT) QUAD --- #

res_numint_joint_quad = numint_joint_quad(loss_function, dist_YX, sol_tol=1e-3, k_sd=4)
print(f"Numerical Integration (Joint) dblquad: {res_numint_joint_quad:.4f}")


####################################################
# --- (2B) NUMERICAL INTEGRATION (JOINT) TRAPZ --- #

# Run the integral
di_args_joint = {'loss':loss_function, 'dist_joint':dist_YX}
res_numint_joint_trapz = numint_joint_trapz(**di_args_joint, use_grid=True)
assert numint_joint_trapz(**di_args_joint, use_grid=False) == res_numint_joint_trapz, 'use_grid should not change results'
print(f"Numerical Integration (Joint) trapezoidal: {res_numint_joint_trapz:.4f}")
# Compate run-time
t_grid = timeit("numint_joint_trapz(**di_args_joint, use_grid=False)", number=num_t, globals=globals())
t_loop = timeit("numint_joint_trapz(**di_args_joint, use_grid=True)", number=num_t, globals=globals())
print(f'Run time (s): grid={t_grid:.2f}, loop={t_loop:.2f} for joint trapezoidal')


###################################################
# --- (2C) NUMERICAL INTEGRATION (COND) TRAPZ --- #


# (1) CREATE INTO CUSTOM FUNCTIONS (DOCSTRINGS FOR EVERYONE!)
# (2) RE-GENERALIZE TO DIST_X OUTSIDE OF THE BVN (I.E. IF F_X IS A MULTIVARIATE), SINCE BVN ONLY RELEVANT WHEN X ~ MVN, AND F_THETA(X): X'THETA0


# Note that calling quad(outer_integrad, ...), outer_integrad = lambda x: quad(inner_intergrand,...) just takes WAY TOO LONG!! 

# Remember that X | Y = y ~ N(mu_X + sig_X/sig_Y*rho*(y-mu_Y), sig2_X*(1-rho^2))
#   The inner integral will be solving I_X(y) = \int_X ell(Y=y, x)*f_{X|Y}(x,Y=y) dx
#   In other words, the integral says, hey, once we know the value of Y, we plug this into out loss function and conditional density, and integrate over the domain of X
#   Next, we have numerical values of f(y) = I_X(y), so we simply integrate of the domain of Y:
#   \int_Y f(y) dy = \int_Y \int_X I(y, x) dy dx

# Run the integral
di_args_cond = {'loss':loss_function, 'dist_X_uncond': dist_X, 'dist_Y_condX': dist_Yx}
res_numint_cond_trapz = numint_cond_trapz(**di_args_cond, use_grid=True)
assert numint_cond_trapz(**di_args_cond, use_grid=False) == res_numint_cond_trapz, 'use_grid should not change results'
print(f"Numerical Integration (Conditional) trapezoidal: {res_numint_cond_trapz:.4f}")
# Compate run-time
t_grid = timeit("numint_cond_trapz(**di_args_cond, use_grid=False)", number=int(num_t/2), globals=globals())
t_loop = timeit("numint_cond_trapz(**di_args_cond, use_grid=True)", number=int(num_t/2), globals=globals())
print(f'Run time (s): grid={t_grid:.2f}, loop={t_loop:.2f} for conditional trapezoidal')

###############################
# --- (3) COMBINE RESULTS --- #

di_res = {
            'MCI Joint': res_mci_joint,
            'MCI Conditional': res_mci_cond,
            'NumInt Joint Quad': res_numint_joint_quad, 
            'NumInt Joint Trapz': res_numint_joint_trapz,
            'NumInt Conditional Trapz': res_numint_cond_trapz,
        }
df_res = pd.DataFrame.from_dict(di_res, orient='index').\
    rename(columns={0:'integral'}).\
    rename_axis('method').\
    round(3).\
    reset_index()
print(df_res)

