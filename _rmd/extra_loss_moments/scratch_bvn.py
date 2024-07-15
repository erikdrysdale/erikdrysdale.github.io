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
from _rmd.extra_loss_moments._mci import MonteCarloIntegration
from _rmd.extra_loss_moments._numerical import NumericalIntegrator

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
kwargs_bvn_joint = {'loss':loss_function, 'dist_YX':dist_YX}
kwargs_bvn_cond = {'loss':loss_function, 'dist_Y_X':dist_Yx, 'dist_X':dist_X}
bvn_joint = bvn_integral(**kwargs_bvn_joint)
bvn_cond = bvn_integral(**kwargs_bvn_cond)
kwargs_base_joint = {'loss':loss_function, 'dist_joint':dist_YX}
kwargs_base_cond = {'loss':loss_function, 'dist_Y_condX':dist_Yx, 'dist_X_uncond':dist_X}
mci_joint = MonteCarloIntegration(**kwargs_base_joint)
mci_cond = MonteCarloIntegration(**kwargs_base_cond)
numint_joint = NumericalIntegrator(**kwargs_base_joint)
numint_cond = NumericalIntegrator(**kwargs_base_cond)


#######################################
# --- (2) MONTE CARLO INTEGRATION --- #

# How many samples to draw frowm
num_samples = 2000000
seed = 12345

# -- (A) Joint distribution -- #
di_args_mci = {'num_samples': num_samples, 'seed': seed, 'calc_variance': True, 'n_chunks': 1}
mu_mci_joint, var_mci_joint = mci_joint.integrate(**di_args_mci)
print(f"Monte Carlo Integration (joint): mean={mu_mci_joint:.3f}, variance={var_mci_joint:.3f}")

# -- (B) Conditional distribution -- #
mu_mci_cond, var_mci_cond = mci_cond.integrate(**di_args_mci)
print(f"Monte Carlo Integration (conditional): mean={mu_mci_cond:.3f}, variance={var_mci_cond:.3f}")

#####################################
# --- (3) NUMERICAL INTEGRATION --- #

# -- (A) Joint quadrature -- #
di_args_joint_quad = {'method': 'quadrature', 'calc_variance': True,
                      'sol_tol': 1e-3, 'k_sd': 4}
mu_numint_joint_quad, var_numint_joint_quad = numint_joint.integrate(**di_args_joint_quad)
print(f"Numerical Integration (Joint) dblquad: mean={mu_numint_joint_quad:.3f}, variance={var_numint_joint_quad:.3f}")

# -- (B) Joint trapezoidal -- #
di_args_joint_trapz = {**di_args_joint_quad, **{'method': 'trapz_loop'}}
mu_numint_joint_trapz, var_numint_joint_trapz = numint_joint.integrate(**di_args_joint_trapz)
print(f"Numerical Integration (Joint) trapezoidal: {mu_numint_joint_trapz:.3f}, variance={var_numint_joint_trapz:.3f}")

# -- (C) Conditional trapezoidal -- #
di_args_cond_trapz = di_args_joint_trapz.copy()
mu_numint_cond_trapz, var_numint_cond_trapz = numint_cond.integrate(**di_args_cond_trapz)
print(f"Numerical Integration (Conditional) trapezoidal: {mu_numint_cond_trapz:.3f}, variance={var_numint_cond_trapz:.3f}")

######################
# --- (4) CHECKS --- #

# (i) Check the monte carlo equivalencies
res_mci_joint = np.array([mu_mci_joint, var_mci_joint])
import sys; sys.exit('stopping')
breakpoint()
mci_joint_v2 = np.array(bvn_joint.calculate_risk(method='mci_joint', **di_args_mci))
np.testing.assert_equal(mci_joint_v2, res_mci_joint)
# Repeat for MCI conditional
res_mci_cond = np.array([mu_mci_cond, var_mci_cond])
mci_cond_v2 = np.array(bvn_cond.calculate_risk(method='mci_cond', **di_args_mci))
np.testing.assert_equal(mci_cond_v2, res_mci_cond)

# (ii) Check the numerical
numint_joint_quad_v2 = np.array(bvn_joint.calculate_risk(method='numint_joint_quad', **di_args_joint_quad))
res_joint_quad = np.array([mu_numint_joint_quad, var_numint_joint_quad])
np.testing.assert_equal(res_joint_quad, numint_joint_quad_v2)
# Repeat for joint traps
numint_joint_trapz_v2 = np.array(bvn_joint.calculate_risk(method='numint_joint_trapz', **di_args_joint_trapz))
res_joint_trapz = np.array([mu_numint_joint_trapz, var_numint_joint_trapz])
np.testing.assert_equal(res_joint_trapz, numint_joint_trapz_v2)
# Repeat for conditional trapz
numint_cond_trapz_v2 = np.array(bvn_cond.calculate_risk(method='numint_cond_trapz', **di_args_cond_trapz))
res_cond_trapz = np.array([mu_numint_cond_trapz, var_numint_cond_trapz])
np.testing.assert_equal(res_cond_trapz, numint_cond_trapz_v2)

# (iii) Check the use_grid arguments
np.testing.assert_equal(numint_joint.integrate(**di_args_joint_trapz, use_grid=False), 
                        res_joint_trapz, 'use_grid should not change results for the joint trapz')
np.testing.assert_equal(numint_cond.integrate(**di_args_cond_trapz, use_grid=False),
                        res_cond_trapz, 'use_grid should not change results for the cond trapz')


#########################
# --- (5) RUN TIMES --- #

# Joint Trapezoidal
t_grid_joint = timeit("numint_joint.integrate(**di_args_joint_trapz, use_grid=True)", number=num_t, globals=globals())
t_loop_joint = timeit("numint_joint.integrate(**di_args_joint_trapz, use_grid=False)", number=num_t, globals=globals())
print(f'Run time (s): grid={t_grid_joint:.2f}, loop={t_loop_joint:.2f} for joint trapezoidal')


t_grid = timeit("numint_cond.integrate(**di_args_cond_trapz, use_grid=True)", number=int(num_t/8), globals=globals())
t_loop = timeit("numint_cond.integrate(**di_args_cond_trapz, use_grid=False)", number=int(num_t/8), globals=globals())
print(f'Run time (s): grid={t_grid:.2f}, loop={t_loop:.2f} for conditional trapezoidal')


###############################
# --- (6) COMBINE RESULTS --- #

di_res = {
            'MCI Joint': res_mci_joint,
            'MCI Conditional': res_mci_cond,
            'NumInt Joint Quad': res_joint_quad, 
            'NumInt Joint Trapz': res_joint_trapz,
            'NumInt Conditional Trapz': res_cond_trapz,
        }
df_res = pd.DataFrame.from_dict(di_res, orient='index').\
    rename(columns={0:'risk', 1:'loss_variance'}).\
    rename_axis('method').\
    round(4).\
    reset_index()
print(df_res)

