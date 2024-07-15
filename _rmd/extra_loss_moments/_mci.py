"""
Utility scripts for monte carlo integration
"""

# External modules
import numpy as np
from typing import Tuple, Union
# Intenral modules
from .utils import BaseIntegrator

class MonteCarloIntegration(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Monte Carlo Integration class. See utils.BaseIntegrator for constructor arguments.
        """
        super().__init__(*args, **kwargs)

    def _draw_samples(self, num_samples: int, seed: int | None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method for generating labels and features

        If has_joint:
            (y_i, x_i) ~ F_{Y, X}
        If ~has_joint:
            X_i \sim F_X
            Y_i | X_i \sim F_{Y | X}
        """
        if self.has_joint:
            y_samples, x_samples = self.dist_joint.rvs(num_samples, random_state=seed).T
        else:
            x_samples = self.dist_X_uncond.rvs(num_samples, random_state=seed)
            y_samples = self.dist_Y_condX(x_samples).rvs(num_samples, random_state=seed+1)
        return y_samples, x_samples

    def _integrate(self, num_samples, calc_variance, seed):
        """
        Internal method for calculating self.integrate() for an arbitrary number of chunks
        """
        # Draw samples with the pre-defines method
        y_samples, x_samples = self._draw_samples(num_samples, seed)
        # Calculate the losses
        losses = self.loss(y=y_samples, x=x_samples)
        res = np.mean(losses)
        if calc_variance:
            # Add on variance if requested
            var = np.var(losses, ddof=1)
            res = (res, var)
        return res

    def integrate(self, 
                  num_samples : int, 
                  calc_variance : bool = False, 
                  n_chunks : int = 1,
                  seed : int | None = None
                  ) -> Union[float, Tuple[float, float], np.ndarray]:
        """
        Compute the integral using Monte Carlo integration.

        Parameters
        ----------
        num_samples : int
            How many samples to draw: dist_joint.rvs(num_samples, ...)?
        calc_variance : bool, optional
            Should the variance be returned (in addition to the mean)?
        n_chunks : int, optional
            Should the Monte Carlo sampling be repeated n_chunks times?
        seed: int | None, optional
            Reproducibility seed
        
            
        Returns
        -------
        float or tuple or np.ndarray
            If ~calc_variance returns \hat{R}
            If calc_variance, then returns (\hat{R}, \hat{V})
            If n_chunks > 0, returns either a 
            \hat{R} = mean([loss(y_1, x_1), ..., loss(y_{num_samples}, x_{num_samples})])
            \hat{V} = var([loss(y_1, x_1), ..., loss(y_{num_samples}, x_{num_samples})])
        """
        # Input checks
        assert n_chunks >= 1, f'n_chunks needs to be >=1, not {n_chunks}'
        # Run calculations
        res = [self._integrate(num_samples, calc_variance, seed=seed+chunk-1) for chunk in range(1,n_chunks+1)]
        res = np.vstack(res).mean(axis=0)
        res = self._return_tuple_or_float(res)
        return res


