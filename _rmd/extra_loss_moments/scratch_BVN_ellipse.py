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
# Internal modules
from .utils import generate_ellipse_points

dir_figs = os.path.join(os.getcwd(), '_rmd', 'extra_loss_moments')
os.listdir(dir_figs)

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



