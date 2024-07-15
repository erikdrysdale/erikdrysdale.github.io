"""
Main functions to support risk and loss variance

python3 -m _rmd.extra_loss_moments.loss
"""

# External modules
import numpy as np
from inspect import signature
from typing import Callable, Union
# Intenral modules
from .utils import accepted_dists
from ._mci import mci_joint, mci_cond
from ._numerical import numint_joint_trapz, numint_cond_trapz, numint_joint_quad

# Shared parameters
hint_DistributionTypes = Union[accepted_dists + (None,)]
# List of the different methods
di_methods = {
    'numint_joint_trapz': 
        {'name': 'Numerical Integration (Joint): Trapezoidal',
         'method': numint_joint_trapz},
    'numint_cond_trapz': 
        {'name': 'Numerical Integration (Conditional): Trapezoidal',
        'method': numint_cond_trapz},
    'numint_joint_quad': 
        {'name': 'Numerical Integration (Joint): Quadrature',
        'method': numint_joint_quad},
    'mci_joint': 
        {'name': 'Monte Carlo Integration (Joint)',
        'method': mci_joint},
    'mci_cond': 
        {'name': 'Monte Carlo Integration (Conditional)',
        'method': mci_cond},
}
valid_methods = list(di_methods.keys())
method_desc = '\n'.join([f"'{k}': {v['name']}" for k, v in di_methods.items()])
print(method_desc)


class bvn_integral():
    def __init__(self,
                loss : Callable,
                dist_YX : hint_DistributionTypes = None,
                dist_Y_X : hint_DistributionTypes = None,
                dist_X : hint_DistributionTypes = None,
                ) -> None:
        """
        Workhorse class for calculate the risk and loss variance of:

        R = E[loss(Y,X)]
        V = E[(loss(Y,X) - E[loss(Y,X)])^2]

        When (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])
        
        Parameters
        ----------
        """
        # Check input construction
        self._check_loss_and_dists(loss=loss, dist_YX=dist_YX, dist_Y_X=dist_Y_X, dist_X=dist_X)
        # Assign as attributes
        self.loss = loss
        self.dist_YX = dist_YX
        self.dist_Y_X = dist_Y_X
        self.dist_X = dist_X

    def calculate_risk(self, 
            method : str,
            k_sd : float | None = None,
            n_Y : int | None = None,
            n_X : int | None = None,
            use_grid : bool | None = None,
            sol_tol : float | None = None,
            num_samples : int | None = None,
            seed : int | None = None,
            **kwargs,
            ) -> float | np.ndarray:
        f"""
        Main method to calculate the risk of a bivariate normal risk and loss variance for a valid method

        Parameters
        ----------
        method : str 
            'numint_joint_trapz': Numerical Integration (Joint): Trapezoidal
            'numint_cond_trapz': Numerical Integration (Conditional): Trapezoidal
            'numint_joint_quad': Numerical Integration (Joint): Quadrature
        **kwargs
            Other named arguments for the various methods

        Returns
        -------
        float | np.ndarray
            A scalar or array of scalars of the risk(s) that have been solved
        """
        # Input checks
        assert method in valid_methods, f'method must be one of {valid_methods}, not {method}'
        integral_method = di_methods[method]['method']
        # Put everything in an argument dictionary
        di_args = {
                    'loss': self.loss, 
                    'dist_joint': self.dist_YX, 
                    'dist_X_uncond': self.dist_X, 
                    'dist_Y_condX': self.dist_Y_X,
                    'k_sd': k_sd, 'n_Y': n_Y, 'n_X': n_X,
                    'use_grid': use_grid, 'sol_tol': sol_tol,
                    'num_samples': num_samples, 'seed': seed,
                    }
        di_args = {**kwargs, **di_args}
        di_args = self._subset_args(di_args, integral_method)
        di_args = {k: v for k, v in di_args.items() if v is not None}
        risk = integral_method(**di_args)
        return risk

    @staticmethod
    def _subset_args(di: dict, func: Callable) -> dict:
        """Convenience wrappers to return the keys of a dictionary that have the same named arguments"""
        named_args = signature(func).parameters.keys()
        filtered_dict = {k: di[k] for k in named_args if k in di}
        return filtered_dict

    @staticmethod
    def _check_loss_and_dists(
            loss,
            dist_YX,
            dist_Y_X,
            dist_X
            ) -> bool:
        """
        Makes sure that loss functions and distributions provided to constructed the BVN integral class word as expected.

        Returns
        -------
        bool
            Whether its a joint distribution
        """
        # Check loss function
        assert isinstance(loss, Callable), f'loss must be a callable function, not {type(loss)}'
        is_callable = True
        try:
            _ = loss(y=1, x=1)
        except:
            is_callable = False
        assert is_callable, 'expected to be abe to call loss(y=..., x=...)'
        # Check distributions
        valid_dists = ', '.join([d.__name__ for d in accepted_dists])
        has_YX = dist_YX is not None
        has_Y_X = dist_Y_X is not None
        has_X = dist_X is not None
        if has_YX:
            assert isinstance(dist_YX, accepted_dists), \
                f'dist_YX must be of type(s) {valid_dists}, not {type(dist_YX)}'
            is_joint = True
        else:
            assert has_Y_X and has_X, f'If dist_YX is None, then both Y_X and X must be provided'
            assert isinstance(dist_Y_X, accepted_dists), \
                f'dist_Y_X must be of type(s) {valid_dists}, not {type(dist_Y_X)}'
            assert isinstance(dist_X, accepted_dists), \
                f'dist_X must be of type(s) {valid_dists}, not {type(dist_X)}'
            is_joint = False
        return is_joint
