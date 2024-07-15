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
from _rmd.extra_loss_moments.utils import dist_Ycond
from _rmd.extra_loss_moments.loss import bvn_integral
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
# Construct the BVN class
bvn_joint = bvn_integral(loss = loss_function, dist_YX=dist_YX)
bvn_cond = bvn_integral(loss = loss_function, dist_Y_X=dist_Yx, dist_X=dist_X)


#######################################
# --- (2) MONTE CARLO INTEGRATION --- #

# How many samples to draw frowm
num_samples = 2000000
seed = 12345

# -- (A) Joint distribution -- #
res_mci_joint = mci_joint(loss_function, dist_YX, num_samples, seed)
print(f"Monte Carlo Integration (joint): {res_mci_joint:.4f}")

# -- (B) Conditional distribution -- #
res_mci_cond = mci_cond(loss_function, dist_X, dist_Yx, num_samples, seed)
print(f"Monte Carlo Integration (conditional): {res_mci_cond:.4f}")


#####################################
# --- (3) NUMERICAL INTEGRATION --- #

# -- (A) Joint quadrature -- #
di_args_joint_quad = {'loss': loss_function, 'dist_joint': dist_YX, 
                      'sol_tol': 1e-3, 'k_sd': 4}
res_numint_joint_quad = numint_joint_quad(**di_args_joint_quad)
print(f"Numerical Integration (Joint) dblquad: {res_numint_joint_quad:.4f}")

# -- (B) Joint trapezoidal -- #
di_args_joint_trapz = {'loss':loss_function, 'dist_joint':dist_YX}
res_numint_joint_trapz = numint_joint_trapz(**di_args_joint_trapz, use_grid=True)
print(f"Numerical Integration (Joint) trapezoidal: {res_numint_joint_trapz:.4f}")

# -- (C) Conditional trapezoidal -- #
di_args_cond_trapz = {'loss':loss_function, 'dist_X_uncond': dist_X, 'dist_Y_condX': dist_Yx}
res_numint_cond_trapz = numint_cond_trapz(**di_args_cond_trapz, use_grid=True)
print(f"Numerical Integration (Conditional) trapezoidal: {res_numint_cond_trapz:.4f}")


######################
# --- (4) CHECKS --- #

# (i) Check the monte carlo equivalencies
# res_mci_joint_v2 = bvn_joint.calculate_risk(method='...', ...)
# np.testing.assert_equal(res_mci_joint, res_mci_joint_v2)
# res_mci_cond_v2 = bvn_joint.calculate_risk(method='...', ...)
# np.testing.assert_equal(res_mci_cond, res_mci_cond_v2)

# (ii) Check the numerical
res_numint_joint_quad_v2 = bvn_joint.calculate_risk(method='numint_joint_quad', **di_args_joint_quad)
np.testing.assert_equal(res_numint_joint_quad, res_numint_joint_quad_v2)
res_numint_joint_trapz_v2 = bvn_joint.calculate_risk(method='numint_joint_trapz', **di_args_joint_trapz)
np.testing.assert_equal(res_numint_joint_trapz, res_numint_joint_trapz_v2)
res_numint_cond_trapz_v2 = bvn_cond.calculate_risk(method='numint_cond_trapz', **di_args_cond_trapz)
np.testing.assert_equal(res_numint_cond_trapz, res_numint_cond_trapz_v2)

# (iii) Check the use_grid arguments
assert numint_joint_trapz(**di_args_joint_trapz, use_grid=False) == res_numint_joint_trapz, 'use_grid should not change results'
assert numint_cond_trapz(**di_args_cond_trapz, use_grid=False) == res_numint_cond_trapz, 'use_grid should not change results'


#########################
# --- (5) RUN TIMES --- #

# Joint Trapezoidal
t_grid_joint = timeit("numint_joint_trapz(**di_args_joint_trapz, use_grid=False)", number=num_t, globals=globals())
t_loop_joint = timeit("numint_joint_trapz(**di_args_joint_trapz, use_grid=True)", number=num_t, globals=globals())
print(f'Run time (s): grid={t_grid_joint:.2f}, loop={t_loop_joint:.2f} for joint trapezoidal')


t_grid = timeit("numint_cond_trapz(**di_args_cond_trapz, use_grid=False)", number=int(num_t/2), globals=globals())
t_loop = timeit("numint_cond_trapz(**di_args_cond_trapz, use_grid=True)", number=int(num_t/2), globals=globals())
print(f'Run time (s): grid={t_grid:.2f}, loop={t_loop:.2f} for conditional trapezoidal')


###############################
# --- (6) COMBINE RESULTS --- #

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

