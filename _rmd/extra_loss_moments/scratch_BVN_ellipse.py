"""
Check Gaussian Bivariate Ellipse calculations

python3 -m _rmd.extra_loss_moments.scratch_BVN_ellipse
"""

# External modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats

dir_figs = os.path.join(os.getcwd(), '_rmd', 'extra_loss_moments')
os.listdir(dir_figs)

###############################
# --- (1) CREATE FUNCTION --- #

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


#######################################
# --- (2) RUN AND PLOT SIMULATION --- #

# Set up parameters
mu_Y = 1.4
mu_X = -0.5
sigma2_Y = 2.1
sigma2_X = 0.9
rho = 0.5
mu = [mu_Y, mu_X]
sigma_y = np.sqrt(sigma2_Y)
sigma_x = np.sqrt(sigma2_X)
rho = rho
cov_matrix = [[sigma_y**2, rho * sigma_y * sigma_x],
              [rho * sigma_y * sigma_x, sigma_x**2]]
cov_matrix = np.array(cov_matrix)
dist_mvn = stats.multivariate_normal(mean=mu, cov=cov_matrix)

# Plot random data and ellipses
data = pd.DataFrame(dist_mvn.rvs(100, 1), columns=['X1', 'X2'])
alphas = [0.05, 0.25, 0.5]
ellipses = pd.concat([generate_ellipse_points(dist=dist_mvn, alpha=1-alpha, n_points=25,ret_df=True).assign(alpha=alpha) for alpha in alphas])
gg = (pn.ggplot(data, pn.aes(x='X2', y='X1')) + pn.theme_bw() + 
      pn.geom_point() + 
      pn.labs(x='X', y='Y') + 
      pn.scale_color_discrete(name='Quantile') + 
      pn.geom_path(pn.aes(color='alpha.astype(str)'), data=ellipses)
      )
gg.save(os.path.join(dir_figs, 'mvn_ellipse.png'), height=4, width=6)



